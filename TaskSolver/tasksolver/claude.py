import anthropic
from .common import TaskSpec, ParsedAnswer, Question
from .exceptions import GPTOutputParseException, GPTMaxTriesExceededException
import threading
from typing import List, Tuple, Union
from loguru import logger
from copy import deepcopy
import time
import os

class ClaudeModel(object):
    def __init__(self, api_key:str,
                 task:TaskSpec,
                 model:str = "claude-3-haiku-20240307"):

        self.claude_key:str = api_key
        self.task:TaskSpec = task
        self.model:str = model

    def ask(self,  payload:dict, n_choices=1) -> Tuple[List[dict], List[dict]]:
        """
        args: 
            payload: json dictionary, prepared by `prepare_payload`
        """

        def claude_thread(idx, payload, results):

            # creation of payload
            mod_payload = deepcopy(payload)

            try:
                raw_response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    #messages=[{"role": "user", "content": "Hello, Claude, tell me a number between 1 to 10000 please."}],
                    messages = [mod_payload["messages"]],
                    max_tokens=mod_payload["max_tokens"],
                )
            except Exception as e:
                raise e

            response = raw_response.dict()
            response['content'] = response['content'][0]['text']
            message = {key: response[key] for key in ['role', 'content']}
            metadata = response.copy() # okay
            del metadata["content"]
            results[idx] = {"message": message, "metadata": metadata} 
            return

        client = anthropic.Anthropic(api_key = self.claude_key)

        assert n_choices >= 1
        results = [None]  * n_choices 
        if n_choices > 1:
            claude_jobs = [threading.Thread(target=claude_thread,
                              args=(idx, payload, results))
                                for idx in range(n_choices)]
            for job in claude_jobs:
                job.start()
            for job in claude_jobs:
                job.join()
        else:
            claude_thread(0, payload, results)
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

        content = []
        dic_list = question.get_json()
        for dic in dic_list:
            # The case of text
            if dic['type'] == 'text':
                content.append(dic)

            # The case of vision input
            elif dic['type'] == 'image_url':
                base64enc_image = dic['image_url']['url'].split(',')[1]
                if base64enc_image.startswith("/9j/"):
                    image_format = 'jpeg'
                elif base64enc_image.startswith("iVBORw0KGgo"):
                    image_format = "png"
                elif base64enc_image.startswith("R0lGOD") or base64enc_image.startswith("R0lGOD"):
                    image_format =  "gif"
                elif base64enc_image.startswith("UklGR"):
                    image_format =  "webp"
                else:
                    raise ValueError("Unknown format")

                modified_dic = {
                    'type' : "image",
                    'source' : {
                        'type' : "base64",
                        'media_type' : f"image/{image_format}",
                        'data' : base64enc_image
                    }
                }

                content.append(modified_dic)

        payload = {
            "messages": {
                'role': 'user',
                'content': content
            },
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