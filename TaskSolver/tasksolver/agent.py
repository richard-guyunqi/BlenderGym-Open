"""
General agents class
"""

from .common import *
from .gpt4v import *
from .ollama import *
from .claude import *
from .gemini import *
from .qwen import *
from .phi import *
from .llama import *
from .minicpm import *
from .intern import *
from abc import abstractmethod
from typing import Union, Dict
from bson import ObjectId
from .event import *
from .keychain import KeyChain
import time

import pickle

class Agent(object):
    def __init__(self, api_key:Union[str, KeyChain], task:TaskSpec,
                 vision_model:str="gpt-4-vision-preview",
                 followup_func=None,
                 session_token=None): 
        """
        Args:
            api_key: openAI/Claude api key
            task: Task specification for this agent
            vision_model: string identifier to the vision model used.
        """
        self.followup_func = followup_func 
        self.api_key = api_key # if this is a string, then 
        self.vision_model = vision_model
        self.task = task
        
        if vision_model in ('gpt-4-vision-preview', 'gpt-4', 'gpt-4-turbo', 'gpt-4o-mini', "gpt-4o", "o1-preview", "o1-mini", 'o3-mini', 'o1'):
            # using the open ai key.
            logger.info(f"creating GPT-based agent of type: {vision_model}")
            if isinstance(api_key, KeyChain):
                api_key = api_key["openai"]
            self.visual_interface = GPTModel(api_key, task, model=vision_model)
        
        elif vision_model in ("claude-3-5-sonnet-latest", "claude-3-haiku-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest", 'claude-3-7-sonnet-latest'):
            # using the claude key.
            logger.info(f"creating Claude-based agent of type: {vision_model}")
            if isinstance(api_key, KeyChain):
                api_key = api_key["claude"]
            self.visual_interface = ClaudeModel(api_key, task)
        
        elif vision_model in ('gemini-pro', 'gemini-pro-vision', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'):
            # using the gemini key.
            if isinstance(api_key, KeyChain):
                api_key = api_key["gemini"]
            logger.info(f"creating Gemini-based agent of type: {vision_model}")
            self.visual_interface = GeminiModel(api_key=api_key, task=task, model=vision_model)

        elif vision_model == 'qwen':
            logger.info(f"creating Qwen-based agent of type: Qwen/Qwen2-VL-7B-Instruct-AWQ.")
            self.visual_interface = QwenModel(task=task, model='Qwen/Qwen2-VL-7B-Instruct-AWQ')

        elif vision_model == 'phi':
            logger.info(f"creating Phi-based agent of type: microsoft/Phi-3.5-vision-instruct.")
            self.visual_interface = PhiModel(task=task, model='microsoft/Phi-3.5-vision-instruct')
            
        elif vision_model == 'llama':
            logger.info(f"creating LLaMA-based agent of type: meta-llama/Meta-Llama-3.1-8B-Instruct.")
            self.visual_interface = LlamaModel(task=task, model='meta-llama/Meta-Llama-3.1-8B-Instruct')

        elif vision_model == 'minicpm':
            logger.info(f"creating MiniCPM-based agent of type: openbmb/MiniCPM-V-2_6-int4.")
            self.visual_interface = MiniCPMModel(task=task, model='openbmb/MiniCPM-V-2_6-int4')

        elif vision_model == 'intern':
            logger.info(f"creating Intern-based agent of type: OpenGVLab/InternVL2-8B.")
            self.visual_interface = InternModel(task=task, model='OpenGVLab/InternVL2-8B')

        '''
        # # TODO: Add your own model here
        # elif vision_model == "{model_id of your model}":
        #     logger.info(f"creating {Name of your model}-based agent of type: {vision_model}")
        #     self.visual_interface = YourModel(task=task, model=vision_model)
        '''
            
        else:
            raise ValueError('Model id not matched with any avalable choices.')
         
        if session_token is None:
            self.session_token = str(ObjectId())
            self.event_buffer = EventCollection()
        else:
            raise NotImplementedError("Need to implement loading function for session_token")

    def save(self, to):
        with open(to, "wb") as f:
            pickle.dump(self, f)
        return self

    @staticmethod
    def load(fp):
        with open(fp, "rb") as f:
            agent = pickle.load(f)
        return agent

    def clear_event_buffer(self):
        # begins a new session, fresh session id and event_buffer objects.
        self.session_token = str(ObjectId())
        self.event_buffer = EventCollection()

    def think(self, question:Question) -> ParsedAnswer:
        """ 
        Adds a THINKING event to the event buffer.
        
        Args:
            question: The question/task instance we seek to solve.
        """

        # make an initial guess if this is going to be the first try
        if len(self.event_buffer.filter_to('ACT')) == 0: 
            p_ans, ans, meta, p = self.visual_interface.run_once(question)
        else:
            print('Into think')
            p_ans, ans, meta, p = self.visual_interface.rough_guess(question)

        ev = ThinkEvent(session_token=self.session_token, 
                        qa_sequence=[(question, p_ans)]) 
        self.event_buffer.add_event(ev)
    
        # update events_collection
        return p_ans, ans, meta, p 
        

    @abstractmethod 
    def act(self, p_ans:ParsedAnswer):
        """
        NEEDS to add an ACTION event to the event buffer.
        
        Executes the action within the environment, resulting
        in some state change.
        This code is specific to the environment/task that it operates under.
        """
        ...


    @abstractmethod
    def observe(self, state:dict):
        """ Observations 
        NEEDS to add an OBSERVE event to the event buffer.
        
        States are specific to the environment/task that it operates under.
        """ 
        ...


    def reflect(self) -> Union[None, Question]:
        """ Reflections
        Adds a REFLECT event to the event buffer.        
        """

        # have we finished the task?

        # evaluator fucntion (self.task.completed) gets the agent itself.
        evaluation_question, evaluation_answer = self.task.completed(self)
        ev = EvaluateEvent(completion_question=evaluation_question,
                         completion_eval=evaluation_answer)
        logger.info(f"evaluator says: {evaluation_answer.success()} -- {evaluation_answer}")
        self.event_buffer.add_event(ev)
        if evaluation_answer.success():
            return None

        # followup func should take in the agent itself,
        # with access to all the events and internal states
        # that it contains, and ask good followup questions
        # to itself. 
        followup = self.followup_func(self)
        ev = FeedbackEvent(feedback=followup)
        self.event_buffer.add_event(ev)
        # otherwise  make the followup. 
        return followup

    def interject(self, interjection:InteractEvent):
        """ User interjects.
        Adds a INTERACT event to the event buffer
        
        Main responsibility of method is storage of 
        user interactions.
        Composed of:
            1) User actions
            2) State transitions
            3) Reasoning, and/or comments for why the agents
               has failed.
        """
        self.event_buffer.add_event(interjection)
        return self        

    def run(self):
        """ An interface to run the T/A/O/R/I loops
        T = think
        A = act
        O = observe
        R = reflect
        I = interaction/interjection
        
        A usual flow over the different steps might look something
        like: TAORTAORTAORTAORI, with an interjection at the end
        from the user as a way to teach the agent how to do the right 
        thing, as well as explanations for why.
        """

        raise NotImplementedError


   