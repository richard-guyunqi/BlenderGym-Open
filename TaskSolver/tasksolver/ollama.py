import ollama
from .common import TaskSpec, ParsedAnswer, Question
from .exceptions import GPTOutputParseException, GPTMaxTriesExceededException
import threading
from typing import List, Tuple, Union
from loguru import logger
from copy import deepcopy
import time

class OllamaModel(object):
    def __init__(self, 
                 task:TaskSpec,
                 model:str):
        self.task:TaskSpec = task
        self.model:str = model

    def ask(self,  payload:dict, n_choices=1) -> Tuple[List[dict], List[dict]]:
        """
        args: 
            payload: json dictionary, prepared by `prepare_payload`
        """

        def ollama_thread(idx, payload, results):

            # creation of payload
            string_message = "\n".join([el["text"] for el in payload["messages"]["content"]])
            mod_payload = deepcopy(payload)
            mod_payload["messages"]["content"] = string_message # overridding with string version
            #print('string_message: ', string_message)
            #print('mod_payload["messages"]: ', payload["messages"])

            
            try:
                response = ollama.chat(model=self.model, messages=[
                        mod_payload["messages"]])
            except Exception as e:
                raise e
            message = response["message"]
            metadata = response.copy()
            del metadata["message"]
            results[idx] = {"message": message, "metadata": metadata} 
            return

        assert n_choices >= 1
        results = [None]  * n_choices 
        if n_choices > 1:
            ollama_jobs = [threading.Thread(target=ollama_thread,
                              args=(idx, payload, results))
                                for idx in range(n_choices)]
            for job in ollama_jobs:
                job.start()
            for job in ollama_jobs:
                job.join()
        else:
            ollama_thread(0, payload, results)
        messages:List[dict] = [ res["message"] for res in results]
        metadata:List[dict] = [ res["metadata"] for res in results]
        return messages, metadata 

    @staticmethod
    def prepare_payload(question:Question,
            verbose:bool=False,
            prepend:Union[dict, None]=None,
            **kwargs
            ) -> dict:


        payload = {
            "messages": {
                'role': 'user',
                'content': question.get_json()
            },
        }
        
        return payload


    def rough_guess(self, question:Question,
                    max_tries=10, query_id:int=0,
                    verbose=False,
                    **kwargs):
    
        p = self.prepare_payload(question, verbose=verbose, prepend=None, 
                                    model=self.model)

        ok = False
        while not ok:
            response, meta_data = self.ask(p) 
            response = response [0] 
            try: 
                parsed_response = self.task.answer_type.parser(response["content"])
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


    def many_rough_guesses(self, num_threads:int,
                           question:Question, 
                           verbose=False, max_tries=10, 
                           **kwargs) -> List[Tuple[ParsedAnswer, str, dict, dict]]:
        """
        Args:
            num_threads : number of independent threads.
            all other  arguments are same as those of `rough_guess()`

        Returns
            List of elements, each element is a tuple following the
            return signature of `rough_guess()`
        """

        p = self.prepare_payload(question, verbose=verbose, prepend=None, 
                                    model=self.model)

        #  TODO
        n_choices = num_threads

        # TODO: wrap in robust-ask method, repeatedly asks until parseable output. 
        ok = False
        reattempt = 0
        while not ok:
            response, meta_data = self.ask(p, n_choices=n_choices)
            try:
                parsed_response = [self.task.answer_type.parser(r["content"]) for r in response]
            except GPTOutputParseException as e:
                logger.warning(f"The following was not parseable:\n\n{response}\n\nBecause\n\n{e}")

                # TODO provide the parse error message into GPT for the next round to be parsable
                reattempt += 1
                if reattempt > max_tries:
                    logger.error(f"max tries ({max_tries}) exceeded.")
                    raise GPTMaxTriesExceededException
             
                logger.warning(f"Reattempt #{reattempt} querying LLM")
                continue
            ok = True 

        return parsed_response, response, meta_data, p

    def run_once(self, question:Question, **kwargs):
        q = self.task.first_question(question) 
        p_ans, ans, meta, p = self.rough_guess(q, **kwargs)
        return p_ans, ans, meta, p