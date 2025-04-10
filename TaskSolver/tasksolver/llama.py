import argparse
import os

import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import sys

import transformers

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from .gpt4v import TaskSpec, ParsedAnswer, Question
from .exceptions import GPTOutputParseException, GPTMaxTriesExceededException
import threading
from typing import List, Tuple, Union
from loguru import logger
from copy import deepcopy
import time
import os

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False



class LlamaModel(object):
    def __init__(self, task:TaskSpec,
                 model:str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):

        self.task:TaskSpec = task
        self.model_id = model

        num_gpus = torch.cuda.device_count()

        # Assign device based on the available GPUs
        if num_gpus == 2:
            self.device_map = "cuda:1"
        if num_gpus == 1:            
            self.device_map = "cuda:0"
    
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"attn_implementation":"flash_attention_2", "torch_dtype": torch.float16},
            device_map=self.device_map,  # Map the model to GPU 2 (index 1)
        )
        
        # self.pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=self.model_id,
        #     model_kwargs={"torch_dtype": torch.bfloat16},
            # device_map="auto",  # Map the model to GPU 2 (index 1)
        # )

    def ask(self,  payload:dict, n_choices=1, temperature=0.7) -> Tuple[List[dict], List[dict]]:
        """
        args: 
            payload: json dictionary, prepared by `prepare_payload`
        """

        def llama_thread(self, idx, payload, results, temperature):

            # creation of payload
            mod_payload = deepcopy(payload)
            messages = payload['messages']
            max_tokens = payload['max_tokens']


            try:
                with torch.autocast(device_type='cuda'):
                    output_text = self.pipeline(
                        messages,
                        max_new_tokens=max_tokens,
                    )

            except Exception as e:
                raise e
            
            print('outputs: ', output_text)
            message = output_text[0]["generated_text"][-1]

            results[idx] = {"metadata": output_text, "message": message} 

            return

        assert n_choices >= 1
        results = [None]  * n_choices 
        if n_choices > 1:
            llama_jobs = [threading.Thread(target=llama_thread,
                            args=(self, idx, payload, results, temperature))
                                for idx in range(n_choices)]
            for job in llama_jobs:
                job.start()
            for job in llama_jobs:
                job.join()
        else:
            llama_thread(self, 0, payload, results, temperature)

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

        image_dic = None
        text = ''
        dic_list = question.get_json()
        for dic in question.get_json():
            # The case of text
            if dic['type'] == 'text':
                text += dic['text']

            # The case of vision input
            elif dic['type'] == 'image_url':
                image_dic = dic['image']

        payload = {
            "messages": [
                {"role": "user", "content": text},
            ],
            "max_tokens": max_tokens,
        }


        return payload


    def rough_guess(self, question:Question, max_tokens=1000,
                    max_tries=10, query_id:int=0,
                    verbose=False, temperature=1,
                    **kwargs):
    
        p = self.prepare_payload(question, max_tokens = max_tokens, verbose=verbose, prepend=None)

        ok = False
        reattempt = 0
        while not ok:
            response, meta_data = self.ask(p, temperature=temperature) 
            response = response[0] 
            logger.info(f'response: {response}')
            try: 
                parsed_response = self.task.answer_type.parser(response["content"])
            except GPTOutputParseException as e:
                logger.warning(f"The following was not parseable:\n\n{response}\n\nBecause\n\n{e}")

                # if not os.path.exists('errors/'):
                #     # Create the directory if it doesn't exist
                #     os.makedirs('errors/')
                # error_saved = f'errors/{time.strftime("%Y-%m-%d-%H-%M-%S")}.json'
                # with open(error_saved, "w")  as f:
                #     f.write(p_ans.code)
                # logger.warning(f"The following was not parseable. Saved in {error_saved}.")
                
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
                           verbose=False, max_tries=10, temperature=1
                           ) -> List[Tuple[ParsedAnswer, str, dict, dict]]:
        """
        Args:
            num_threads : number of independent threads.
            all other  arguments are same as those of `rough_guess()`

        Returns
            List of elements, each element is a tuple following the
            return signature of `rough_guess()`
        """

        p = self.prepare_payload(question, max_tokens = max_tokens, verbose=verbose, prepend=None)

        #  TODO
        n_choices = num_threads

        # TODO: wrap in robust-ask method, repeatedly asks until parseable output. 
        ok = False
        reattempt = 0
        while not ok:
            response, meta_data = self.ask(p, n_choices=n_choices, temperature=temperature)
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

    def run_once(self, question:Question, max_tokens=1000, temperature=1, **kwargs):
        q = self.task.first_question(question) 
        p_ans, ans, meta, p = self.rough_guess(q, max_tokens=max_tokens, temperature=temperature, **kwargs)
        return p_ans, ans, meta, p





