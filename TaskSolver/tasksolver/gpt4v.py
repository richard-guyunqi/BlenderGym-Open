from typing import List, Union, Tuple
from loguru import logger
import io
from openai import OpenAI
from .common import TaskSpec, ParsedAnswer, Question
from .exceptions import GPTOutputParseException, GPTMaxTriesExceededException


class GPTModel(object):
    def __init__(self, api_key:str,
                 task:TaskSpec, 
                 model:str="gpt-4o",
                 ):
        self.open_ai_key:str = api_key
        
        self.task:TaskSpec = task
        self.model:str = model
 
    def ask(self, payload: dict, n_choices=1) -> Tuple[dict, dict]:
        """
        args:
            payload: json dictionary, prepared by `prepare_payload`
        """
        client = OpenAI(api_key=self.open_ai_key) 

        try:
            if self.model in ("gpt-4o", 'gpt-4o-mini', 'gpt-4-turbo'):
                response = client.chat.completions.create(
                    model=self.model, #"gpt-4-vision-preview",
                    messages=payload["messages"],
                    max_tokens=payload["max_tokens"],
                    n=n_choices
                )
            elif self.model in ('o1-mini', 'o1', 'o3-mini', 'o3'):
                response = client.chat.completions.create(
                    model=self.model, #"gpt-4-vision-preview",
                    messages=payload["messages"],
                    max_completion_tokens=payload["max_tokens"],
                    n=n_choices
                )

            
            
        except Exception as e:
            # err = e
            raise e 
            

        response = response.dict()
        messages = [choice["message"] for choice in response["choices"]]        
        
        metadata = response["usage"]

        return messages, metadata

    @staticmethod
    def prepare_payload(question:Question,
            verbose:bool=False,
            prepend:Union[dict, None]=None,
            model:str="gpt-4-vision-preview",
            max_tokens:int=1000,
            ) -> dict:
        """
        Args: 
            question: List of question components
            verbose: if true, prints out the payload.
            prepend (optional): if not None it should be the "message" from the 
                GPT output from the previous exchange.
        Returns:
            payload (dict) containing the json to be sent to GPT's API.

        """
        question_dicts = question.get_json()
        print('Getting question_dicts fine.')

        for part in question_dicts:
            if part["type"]=="image_url":
                del part["image"] # remove the PIL.Image
             
        payload = [{"role": "user",
                    "content": question_dicts
                    }]
        if prepend is not None:
            payload = [prepend] + payload

        if verbose:
            print("############")
            "\n".join([str(el) for el in payload])
            print("############")

        payload = {
            "model": model,
            "messages": payload,
            "max_tokens": max_tokens}
        return payload

    def run_once(self, question:Question, max_tokens=1000):
        print('Into gpt4v.py -> think')
        q = self.task.first_question(question) 
        p_ans, ans, meta, p = self.rough_guess(q, max_tokens=max_tokens)
        return p_ans, ans, meta, p


    ############### NOTE : deprecated -- moved to Agent class.
    def run(self, question:Question, verbose:bool=False):
        """ Main running program
        """
        logger.warning("DEPRECATED! Use the Agents class instead!")
        
        answers_history = []
        questions_history = []
        eval_history = []

        first_q = self.task.first_question(question) 
        p_ans, ans, meta, p = self.rough_guess(first_q) 
        
        questions_history.append(question)
        answers_history.append(p_ans)
        latest_answer = p_ans

        if verbose:
            logger.info(f"iteration 0 Answer: {str(p_ans)}")

        iteration = 0 
        
        while True: 
            evaluation_answer = self.task.completed(question, latest_answer)
            eval_history.append(evaluation_answer)
            if verbose:
                logger.info(f"eval comment from {iteration} editing: \n {str(evaluation_answer)}")
            
            if evaluation_answer.success():
                break

            iteration += 1
            if verbose:
                logger.info(f"start iteration {iteration} editing")
            next_question = self.task.next_question(questions_history, answers_history, eval_history) 
            p_ans, ans, meta, p = self.rough_guess(next_question)
            
            answers_history.append(p_ans) 
            latest_answer = p_ans
            if verbose:
                logger.info(f"iteration {iteration} editing output: \n{str(p_ans)}")
        if verbose:
            logger.info(f"Returning answer at iteration {iteration}: \n{str(p_ans)}")
        return latest_answer, ans, meta, p


    
    def many_rough_guesses(self, num_threads:int,
                           question:Question, max_tokens=1000, 
                           verbose=False, max_tries=10) -> List[Tuple[ParsedAnswer, str, dict, dict]]:
        """
        Args:
            num_threads : number of independent threads.
            all other  arguments are same as those of `rough_guess()`

        Returns
            List of elements, each element is a tuple following the
            return signature of `rough_guess()`
        """

        p = self.prepare_payload(question, verbose=verbose, prepend=None, 
                                    model=self.model,
                                    max_tokens=max_tokens)

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


    def rough_guess(self, question:Question, max_tokens=1000, verbose=False,
                    max_tries=10, query_id:int=0) -> Tuple[ParsedAnswer, str, dict, dict]:
        """
        Args:
            question
            max_tokens (int) : max tokens in return from
            verbose (bool) 
        Returns:
            answer in the form of ParsedAnswer
            answer in the form of raw text response from LLM
            meta data of the response
            json payload sent to the LLM
        """

        p = self.prepare_payload(question, verbose=verbose, prepend=None, 
                                    model=self.model,
                                    max_tokens=max_tokens)

        print('Loading payload fine.')

        ok = False
        reattempt = 0
        while not ok:
            response, meta_data = self.ask(p)
            response = response[0]
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
            
        