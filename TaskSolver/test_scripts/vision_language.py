"""
Read the speed limit. 

Toy setting for vision-language input to test VLM implementation.
"""

from tasksolver.common import TaskSpec, ParsedAnswer, Question, KeyChain
from tasksolver.ollama import OllamaModel
from tasksolver.llama import LlamaModel
from tasksolver.exceptions import *
from tasksolver.utils import docs_for_GPT4
from tasksolver.claude import ClaudeModel
from tasksolver.gemini import GeminiModel
from tasksolver.qwen import QwenModel
from tasksolver.gpt4v import GPTModel
from tasksolver.phi import PhiModel
from tasksolver.minicpm import MiniCPMModel
from tasksolver.intern import InternModel
from PIL import Image
from pathlib import Path

'''
TODO: Import the class instance for your own model
from tasksolver.your_model import YourModel
'''

api_dict = KeyChain()
api_dict.add_key("openai_api_key", "system/credentials/openai_api.txt")
api_dict.add_key("claude_api_key", "system/credentials/claude_api.txt")
api_dict.add_key("gemini_api_key", "system/credentials/gemini_api.txt")

'''
TODO[optional]: If you are using another model that accepts API queries, add the following
api_dict.add_key("your_api_key", "system/credentials/your_model.txt")
'''

# Load images
image_path = 'TaskSolver/test_scripts/speed_limit.png'
image = Image.open(image_path)

class SpeedLimit(ParsedAnswer):
    def __init__(self, speed_limit:str):
        self.speed_limit = speed_limit

    @staticmethod    
    def parser(gpt_raw:str) -> "ReadSign":
        """
        @GPT4-doc-begin
            ONLY RETURN A NUMBER.
            
                For example,
                
                90

        @GPT4-doc-end
        """

        gpt_out = gpt_raw.strip().strip('.').strip(',').lower()

        if not gpt_out.isdigit():
            raise GPTOutputParseException("output should only contain a number!")

        return SpeedLimit(gpt_out)

    def __str__(self):
        return str(self.speed_limit)
    
read_speed_limit = TaskSpec(
    name="Read Speed Limit",
    description="You are given a picture on the right, which is about a speed limit sign in California . Please read it and find out the exact number of speed limit.",
    answer_type= SpeedLimit,
    followup_func= None,
    completed_func= None
)

read_speed_limit.add_background(
    Question([
        "ONLY RETURN A NUMBER. Read the following for the docs of the parser, which will parse your response, to guide the format of your responses:" , 
        docs_for_GPT4(SpeedLimit.parser) 
    ])
)

if __name__=='__main__':        
    question = Question(["Read the image now. What is the speed limit? ONLY RETURN THE NUMBER.", image])

    # interface = InternModel(task=read_speed_limit)
    interface = ClaudeModel(api_key=api_dict['claude_api_key'], task=read_speed_limit, model='claude-3-5-sonnet-latest')

    '''
    # TODO: add your own model here. 
    # interface = YourModel(task=cointoss)
    # Or if your model requires API:
    # interface = YourModel(api_key=api_dict['your_api_key'], task=cointoss)
    '''

    # Read the sign for a single time
    model_input = read_speed_limit.first_question(question)
    out, _, _, _ = interface.rough_guess(model_input, max_tokens=2000)
    print(out)



