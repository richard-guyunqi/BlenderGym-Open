"""
Heads or tails?

Toy setting for test-only input to test VLM implementation.
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

'''
TODO: Import the class instance for your own model
Example: from tasksolver.your_model import YourModel
'''

api_dict = KeyChain()
api_dict.add_key("openai_api_key", "system/credentials/openai_api.txt")
api_dict.add_key("claude_api_key", "system/credentials/claude_api.txt")
api_dict.add_key("gemini_api_key", "system/credentials/gemini_api.txt")

'''
TODO[optional]: If you are using another model that accepts API queries, add the following
api_dict.add_key("your_api_key", "system/credentials/your_model.txt")
'''

class HoT(ParsedAnswer):
    def __init__(self, heads_or_tails:str):
        self.heads_or_tails = heads_or_tails

    @staticmethod    
    def parser(gpt_raw:str) -> "HoT":
        """
        @GPT4-doc-begin
            gpt_raw: string that contains just a simple "heads" or "tails".
            
                For example,
                
                heads

        @GPT4-doc-end
        """

        gpt_out = gpt_raw.strip().strip('.').strip(',').lower()

        if gpt_out not in ("heads", "tails"):
            raise GPTOutputParseException("output should either be `heads` or `tails`")

        return HoT(gpt_out)

    def __str__(self):
        return str(self.heads_or_tails)
    
cointoss = TaskSpec(
    name="Coin Toss",
    description="You're flipping an unbiased coin (the probability of landing tails is 50\%, and that of landing tails is 50\%) -- output `tails` or `tails` depending on the outcome of the flip.",
    answer_type= HoT,
    followup_func= None,
    completed_func= None
)

cointoss.add_background(
    Question([
        "Read the following for the docs of the parser, which will parse your response, to guide the format of your responses:" , 
        docs_for_GPT4(HoT.parser) 
    ])
)

if __name__=='__main__':
    question = Question(["Toss the coin. What's the outcome?"])

    # interface = InternModel(task=cointoss)
    interface = ClaudeModel(api_key=api_dict['claude_api_key'], task=cointoss, model='claude-3-5-sonnet-latest')

    '''
    # TODO: add your own model here. 
    # interface = YourModel(task=cointoss)
    # Or if your model requires API:
    # interface = YourModel(api_key=api_dict['your_api_key'], task=cointoss)
    '''

    # Toss the coin for a single time
    model_input = cointoss.first_question(question)
    out, _, _, _ = interface.rough_guess(model_input, max_tokens=2000)
    print(out)



