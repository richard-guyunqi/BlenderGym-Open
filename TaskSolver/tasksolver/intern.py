import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import sys
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer

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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        # T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InternModel(object):
    def __init__(self, task:TaskSpec,
                 model:str="OpenGVLab/InternVL2-8B"):

        self.task:TaskSpec = task
        self.model = self.get_model(model)

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)

    
    def get_model(self, model):
        # Load the open-source model in

        if model == "OpenGVLab/InternVL2-8B":
            model_weights = AutoModel.from_pretrained(
                model,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True).eval()
            return model_weights
        else:
            raise ValueError(f"Such model {model} does not exist!")

    def ask(self,  payload:dict, n_choices=1, temperature=0.7) -> Tuple[List[dict], List[dict]]:
        """
        args: 
            payload: json dictionary, prepared by `prepare_payload`
        """

        def intern_thread(self, idx, payload, results, temperature):

            # creation of payload
            mod_payload = deepcopy(payload)
            question = payload['question']
            pixel_values = payload['pixel_values']
            num_patches_list = payload['num_patches_list']
            max_tokens = payload['max_tokens']

            generation_config = dict(max_new_tokens=max_tokens, do_sample=True)

            try:
                # Preparation for inference
                output_text = self.model.chat(self.tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=None)
            
            except Exception as e:
                raise e
            
            print('outputs: ', output_text)
            message = {'content' : output_text}

            results[idx] = {"metadata": output_text, "message": message} 

            return

        assert n_choices >= 1
        results = [None]  * n_choices 
        if n_choices > 1:
            intern_jobs = [threading.Thread(target=intern_thread,
                            args=(self, idx, payload, results, temperature))
                                for idx in range(n_choices)]
            for job in intern_jobs:
                job.start()
            for job in intern_jobs:
                job.join()
        else:
            intern_thread(self, 0, payload, results, temperature)

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
        img_list = []
        for dic in question.get_json():
            # The case of text
            if dic['type'] == 'text':
                text += dic['text']

            # The case of vision input
            elif dic['type'] == 'image_url':
                img_list.append(dic['image'])
                text += '<image>\n'

        pixel_list = [load_image(image).to(torch.float16).cuda() for image in img_list]

        if pixel_list:
            pixel_values = torch.cat(tuple(pixel_list), dim=0)
            num_patches_list = [img_tensor.size(0) for img_tensor in pixel_list]

        else:
            pixel_values = None
            num_patches_list = None

        payload = {
            'question': text,
            'pixel_values': pixel_values,
            'num_patches_list':num_patches_list,
            "max_tokens": max_tokens,
        }

        return payload


    def rough_guess(self, question:Question, max_tokens=1000,
                    max_tries=10, query_id:int=0,
                    verbose=False, temperature=1,
                    **kwargs):
    
        p = self.prepare_payload(question, max_tokens = max_tokens, verbose=verbose, prepend=None, 
                                    model=self.model)

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

    def all_task_rough_guess(self, task, question:Question, max_tokens=1000,
                    max_tries=10, query_id:int=0,
                    verbose=False, temperature=1,
                    **kwargs):
    
        p = self.prepare_payload(question, max_tokens = max_tokens, verbose=verbose, prepend=None, 
                                    model=self.model)

        ok = False
        reattempt = 0
        while not ok:
            response, meta_data = self.ask(p, temperature=temperature) 
            response = response[0] 
            logger.info(f'response: {response}')
            try: 
                parsed_response = task.answer_type.parser(response["content"])
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

        p = self.prepare_payload(question, max_tokens = max_tokens, verbose=verbose, prepend=None, 
                                    model=self.model)

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





