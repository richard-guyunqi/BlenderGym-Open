import os
from .common import TaskSpec, ParsedAnswer, Question
from .exceptions import GPTOutputParseException, GPTMaxTriesExceededException
import threading
import base64
import io
from typing import List, Tuple, Union
from loguru import logger
from google.generativeai.types import generation_types
from copy import deepcopy
import google.generativeai as genai
import time
import PIL

class GeminiModel(object):
    def __init__(self, api_key:str,
                 task:TaskSpec,
                 model:str="gemini-pro-vision"):
        self.gemini_key:str = api_key
        self.task:TaskSpec = task
        self.model:str = model


    def ask(self,  payload:dict, n_choices=1) -> Tuple[List[dict], List[dict]]:
        """
        args: 
            payload: json dictionary, prepared by `prepare_payload`
        """

        def gemini_thread(idx, payload, results):

            mod_payload = payload

            config_instance = generation_types.GenerationConfig(
                max_output_tokens=payload["max_tokens"], 
            )

            try:
                raw_response = client.generate_content(
                    contents=payload["messages"],
                    generation_config=config_instance
                )
            except Exception as e:
                raise e

            response = {'content' : raw_response.text}
            results[idx] = {"message": response, "metadata": raw_response} 
            return

        genai.configure(api_key=self.gemini_key)
        client = genai.GenerativeModel(model_name=self.model,
            safety_settings= None,
            generation_config = None
        )
        
        assert n_choices >= 1
        results = [None]  * n_choices 
        if n_choices > 1:
            gemini_jobs = [threading.Thread(target=gemini_thread,
                              args=(idx, payload, results))
                                for idx in range(n_choices)]
            for job in gemini_jobs:
                job.start()
            for job in gemini_jobs:
                job.join()
        else:
            gemini_thread(0, payload, results)
        messages:List[dict] = [ res["message"] for res in results]
        metadata:List[dict] = [ res["metadata"] for res in results]
        return messages, metadata 


    @staticmethod
    def prepare_payload(question:Question,
            max_tokens=1000,
            verbose:bool=False,
            prepend:Union[dict, None]=None,
            **kwargs
            ) -> dict:

        strings = []
        images = []
        for el in question.get_json(save_local=True):
            if 'text' in el:
                strings.append(el['text'])
            elif 'image_url' in el:
                #Convert the binary encoded version to PIL.image
                base64enc_image = el['image_url']['url'].split(',', 1)[1]
                base64_image_str = base64enc_image  # the Base64-encoded string
                image_data = base64.b64decode(base64_image_str)
                image_data_io = io.BytesIO(image_data)

                # Read the image from the BytesIO object
                pil_image = PIL.Image.open(image_data_io)

                images.append(pil_image)

        string_message = "\n".join(strings)
        messages = [string_message]

        for image in images:
            messages.append(image)

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        return payload


    def rough_guess(self, question:Question, max_tokens=1000,
                    max_tries=10, query_id:int=0,
                    verbose=False,
                    **kwargs):
    
        p = self.prepare_payload(question, max_tokens = max_tokens, verbose=verbose, prepend=None, 
                                    model=self.model)

        ok = False
        while not ok:
            response, meta_data = self.ask(p) 
            response = response [0] 
            try: 
                parsed_response = self.task.answer_type.parser(response["content"])
            except GPTOutputParseException as e:

                if not os.path.exists('errors/'):
                    # Create the directory if it doesn't exist
                    os.makedirs('errors/')

                error_saved = f'errors/{time.strftime("%Y-%m-%d-%H-%M-%S")}.json'
                with open(error_saved, "w")  as f:
                    f.write(p_ans.code)
                logger.warning(f"The following was not parseable. Saved in {error_saved}.")
                
                reattempt += 1
                if reattempt > max_tries:
                    logger.error(f"max tries ({max_tries}) exceeded.")
                    raise GPTMaxTriesExceededException
             
                logger.warning(f"Reattempt #{reattempt} querying LLM")
                continue
            ok = True 

        return parsed_response, response, meta_data, p


    def many_rough_guesses(self, num_threads:int,
                           question:Question, max_tokens=1000,
                           verbose=False, max_tries=10, 
                           ) -> List[Tuple[ParsedAnswer, str, dict, dict]]:
        """
        Args:
            num_threads : number of independent threads.
            all other  arguments are same as those of `rough_guess()`

        Returns
            List of elements, each element is a tuple following the
            return signature of `rough_guess()`
        """

        p = self.prepare_payload(question, max_tokens = max_tokens, verbose=verbose, prepend=None, 
                                    model=self.model)
        #print('In many rough: ', p)

        n_choices = num_threads

        ok = False
        reattempt = 0
        while not ok:
            response, meta_data = self.ask(p, n_choices=n_choices)
            try:
                parsed_response = [self.task.answer_type.parser(r["content"]) for r in response]
            except GPTOutputParseException as e:
                logger.warning(f"The following was not parseable:\n\n{response}\n\nBecause\n\n{e}")

                reattempt += 1
                if reattempt > max_tries:
                    logger.error(f"max tries ({max_tries}) exceeded.")
                    raise GPTMaxTriesExceededException
             
                logger.warning(f"Reattempt #{reattempt} querying LLM")
                continue
            ok = True 

        return parsed_response, response, meta_data, p

    def run_once(self, question:Question, max_tokens=1000, **kwargs):
        q = self.task.first_question(question) 
        p_ans, ans, meta, p = self.rough_guess(q, max_tokens=max_tokens, **kwargs)
        return p_ans, ans, meta, p