import os
from typing import Type, List, Union, Callable, Tuple, TypeVar
from pathlib import Path
import base64
from abc import abstractmethod
from PIL import Image
from loguru import logger
import requests
import io
import threading
from bson import ObjectId
from .utils import URL
import os
from typing import Union, Dict

io_semaphore = threading.Semaphore(1)

T = TypeVar('T', bound="ParsedAnswer")
class ParsedAnswer(object):
    """ Base class to specify parsing output types
    Needs to be specified PER task
    """
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def parser(gpt_raw:str) -> T: 
        # returns an instance of ParsedAnswer
        pass

    @abstractmethod
    def __str__(self):
        pass


class Question(object):
    def __init__(self, elements:Union[None, List[Union[URL, Path, str, ParsedAnswer, Image.Image, 
                        Tuple[Union[URL, Path, str, ParsedAnswer, Image.Image], Union[str, Tuple[str]]]]]]):
        if elements is None:
            self.elements = [] 
        else:
            self.elements = []
            for el in elements:
                if isinstance(el, tuple):
                    assert len(el) == 2
                    if isinstance(el[1], str):
                        el = list(el)
                        el[1] = (el[1],)
                        el = tuple(el)
                elif isinstance( el, list):
                    assert len(el) == 2
                    if isinstance(el[1], str):
                        el[1] = (el[1],)
                else: 
                    el = (el, None)

                # if el[0] is an instance of Question
                if isinstance(el[0], Question):
                    _question_elements = el[0].eval()
                    for qe in _question_elements:
                        if qe[1] == None:
                            tag = el[1] if el[1] is not None else None # give it the tag
                        else:
                            tag = tuple(list(qe[1]) + list(el[1])) if el[1] is not None else qe[1]
                        self.elements.append((qe[0], tag))
                            
                else: 
                    self.elements.append(tuple(el))

    @staticmethod
    def encode_image(image_path:Union[Path, str]) :
        with open(str(image_path), "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def get_text_content(text:str):
        return {"type": "text",
                "text": text}


    @staticmethod
    def get_pil_image_content(image:Image.Image):
        with io_semaphore:
            img_byte_array = io.BytesIO()
            image.save(img_byte_array, format='PNG')  # Save the PIL image to the in-memory stream as PNG
            img_byte_array.seek(0) 
            base64enc_image = base64.b64encode(img_byte_array.read()).decode('utf-8') 
            pack = {"type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64enc_image}"
                    },
                "image": image
                }
        return pack

    @staticmethod
    def get_local_image_content(image_path:Union[Path, str]):
        base64enc_image = Question.encode_image(image_path)
        return {"type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64enc_image}"
                    },
                "image": Image.open(image_path)
                }
      
    @staticmethod     
    def get_pil_image_content_savecopy(image:Image.Image):
        
        directory = "temporary/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # unique id
        unique_id = str(ObjectId())
        filename = f'{unique_id}.jpg'

        # Define the path where the image will be saved
        filepath = os.path.join(directory, filename)
        
        # Save the image with the appropriate format based on the file extension
        image.save(filepath)
        ret = Question.get_local_image_content(filepath)
        ret["local_path"] = filepath
        return ret

    @staticmethod
    def get_remote_image_content(image_url:URL):
        return {"type": "image_url", 
                "image_url": {
                    "url": str(image_url),
                    },
                "image": None # we don't pre-read the image from the url. 
                }

    def __str__(self):
        return "\n".join([str(el[0]) for el in self.elements] )       

    def prepend_question (self, other_question:"Question"):
        self.elements = other_question.eval() + self.elements
        return self
    
    def append_question (self, other_question:"Question"):
        self.elements = self.elements + other_question.eval()
        return self

    def __add__(self, other):
        return Question(self.elements).append_question(other)

    def subquestion(self, filter_tag:Union[Tuple[str], str, None]):
        return Question(self.eval(filter_tag=filter_tag))

    def eval(self, filter_tag:Union[None, Tuple[str], str]=None):
        """ Returns all the question components, and disregards tags
        Args:
            filter_tag: if None, then return everything. Otherwise, return the components that match the
                tags found in filter_tag.
        """
        if isinstance(filter_tag , str): 
            filter_tag = (filter_tag,)
        return_elements = [] 
        for comp, tag in self.elements:
            # NOTE: tag is a tuple or it is None.
            if filter_tag is None:
                # all should be allowed
                return_elements.append((comp, tag))
            elif tag is not None and len(set(tag).intersection(set(filter_tag))) > 0:
                return_elements.append((comp, tag))
            # else, ignore.  (tag = None, or null set intersection)
        return return_elements

    @property
    def question_components(self):
        return [el[0] for el in self.elements]
    
    def images(self) -> List[Image.Image]:
        """ Returns a list of all the images indicated in the Question object.
        """
        imgs = []
        for el in self.eval():
            component = el[0]
            if isinstance(component, Image.Image):
                imgs.append(component)
            elif isinstance(component, Path):
                imgs.append(Image.open(component))
            elif isinstance(component, URL):
                try:
                    response= requests.get(str(component))
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    imgs.append(img)
                except requests.RequestException as e:
                    logger.warning(f"Error fetching the image from URL: {e}")
                    continue
        return imgs                

    def get_json(self, **kwargs): 
        payload = []
        for el in self.question_components:
            if isinstance(el, str):
                payload.append(self.get_text_content(el))
            elif isinstance(el, Image.Image):
                if "save_local" in kwargs and kwargs["save_local"] is True:
                    payload.append(self.get_pil_image_content_savecopy(el))
                else:
                    payload.append(self.get_pil_image_content(el))
            elif isinstance(el, Path):
                payload.append(self.get_local_image_content(el))
            elif isinstance(el, URL):
                payload.append(self.get_remote_image_content(el))
            elif isinstance(el,ParsedAnswer):
                payload.append(self.get_text_content(str(el)))
            else:
                print(self.question_components)
                raise ValueError(f"invalid element type {type(el)} in question input!")       
        return payload 


class TaskSpec(object):
    """ Specifies the task

    Example usage:
        TS = TaskSpec(name="human detector", 
            description="Return the bounding box parameters (lower left, top right) pixel coordinates that circumscribe instances of humans within the input image.") 
        
        # add examples
        TS.add_example(
            input=Question(...), # can include images
            output=ParsedAnswer(...),
            explanation="Because ..." # or None
        )

        # tell it the steps you'd like it to go through to evaluate the problem
        # e.g. takes the input Question, takes the image, and then right-concatenates
        # that with the visualization output of the previous step 
    """
    def __init__(self, 
                 name:str,
                 description:str,
                 answer_type:Type,
                 followup_func:Callable[[List[Question], List[ParsedAnswer]], Question],
                 completed_func:Callable[[Question, ParsedAnswer], bool],
                 ):
        """
        Args:
            followup_func: is a callable that learns maps from the input question, parsed answer
                and returns another question to ask to the model.
        """
        self.name = name
        self.description = description
        
        self.followup_func = followup_func  # custom function, needs to be specified PER Task
        self.completed = completed_func
        self.answer_type = answer_type
        
        self.examples = []
        self.background = None

    def add_background(self, background:Question):
        self.background = background

    def add_example(self, input:Question, output:ParsedAnswer, explanation:Union[str, None]=None):
        """ Used to add examples of I/O to the model.
        """
        assert isinstance(input, Question), "input must be instance of Question"
        assert isinstance(output, ParsedAnswer), "output must be instance of ParsedAnswer"

        self.examples.append({"question": input,
                              "answer": output, 
                              "explanation": explanation})
        return self 

    def task_question_component(self, filter_tag:Union[None, Tuple[str], str]=None):
        return Question([("# Task Description", "TASK_DESC_TITLE"), 
                            (self.description, "TASK_DESC_CONTENT")]).subquestion(filter_tag=filter_tag)
    def background_question_component(self, filter_tag:Union[None, Tuple[str], str]=None):
        assert isinstance(self.background, Question)
        question = Question([("# Background information", "BACKGROUND_TITLE"), 
                            (self.background, "BACKGROUND_CONTENT")])
        return question.subquestion(filter_tag=filter_tag)

    def example_question_component(self, filter_tag:Union[None, Tuple[str], str]=None):
        question = Question([])
        question.append_question(Question([("# Examples", "EXAMPLES_TITLE")]))
        question.append_question(Question([(f"Here are {len(self.examples)} examples:", "EXAMPLES_CONTENT")]))
        for ex_idx, ex_dict in enumerate(self.examples):

            question.append_question(Question([(f"(Ex #{ex_idx}) Question:", ("EXAMPLES_QUESTION_TITLE", f"EXAMPLE_{ex_idx}"))]))
            question.append_question(Question([(ex_dict["question"], ("EXAMPLES_QUESTION_CONTENT", f"EXAMPLE_{ex_idx}"))])) # the question
            
            if ex_dict["explanation"] is not None:
                question.append_question(Question([(f"(Ex #{ex_idx}) Reasoning:", ("EXAMPLES_REASON_TITLE", f"EXAMPLE_{ex_idx}"))]))
                question.append_question(Question([(ex_dict["explanation"], ("EXAMPLES_REASON_CONTENT", f"EXAMPLE_{ex_idx}"))]))
            
            question.append_question(Question([(f"(Ex #{ex_idx}) Answer:", ("EXAMPLES_ANSWER_TITLE", f"EXAMPLE_{ex_idx}"))]))
            question.append_question(Question([(str(ex_dict["answer"]), ("EXAMPLES_ANSWER_CONTENT", f"EXAMPLE_{ex_idx}"))]))
            question.append_question(Question(["\n\n"]))
        return question.subquestion(filter_tag=filter_tag)

    def prompt_question_component(self, user_question, filter_tag:Union[None, Tuple[str], str]=None):
        question = Question([])
        question.append_question(Question([("# Your turn -- please complete the following task:", "QUESTION_TITLE")]))
        question.append_question(Question([(user_question, "QUESTION_CONTENT")]))
        return question.subquestion(filter_tag=filter_tag)

    def first_question(self, question:Question):
        # task description 
        first_q = self.task_question_component()
        # background information 
        if self.background is not None:
            first_q.append_question(self.background_question_component())   
        # examples 
        if len(self.examples) > 0: 
            first_q.append_question(self.example_question_component())        
        # finally, the question 
        first_q.append_question(self.prompt_question_component(question))

        return first_q 
    
    def next_question(self, questions_history:List[Question], 
                            answers_history:List[ParsedAnswer], 
                            eval_history:List[ParsedAnswer]) -> Question:
        next_q = self.followup_func(self, questions_history, answers_history, eval_history)
        return next_q


class KeyChain(object):
    def __init__(self,keys:Union[None, Dict[str, str]]=None):
        if keys is None:
            self.keys = {}
        else:
            assert isinstance(keys, dict), "Keys should be dict."
         
    def add_key(self, service:str, key:str):
        if os.path.exists(key): # it's a file
            with open(key, "r") as f:
                key = f.readline().strip()
        self.keys.update({service: key}) 
        return self

    def get_key(self, service:str ):
        if service not in self.keys:
            raise ValueError(f"No keys associated with '{service}'") 

        return self.keys[service]

    def __getitem__(self, service:str):
        return self.get_key(service)
     