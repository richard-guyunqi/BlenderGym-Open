'''
This is the helper py file to save all kinds of ParsedAnswer type. 

ParsedAnswer: 
    Input: language model raw output(words)
    Output: formatted output according to the given pattern
    Input_instrcution: There will be a few lines that could be imported to the model
        as a in-context guidance
'''


import re
from .common import ParsedAnswer
from .exceptions import *
from typing import List


class LeftOrRight(ParsedAnswer):
    
    def __init__(self, l_or_r:str, gpt_raw:str=None):
        if l_or_r.lower().strip() not in ('left', 'right'):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(gpt_raw)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            raise GPTOutputParseException("output of LLM should either be 'left' or 'right'")
        self.data = l_or_r.lower().strip()
        self.raw = gpt_raw

    @staticmethod
    def remove_answer_text(s:str):
        # Define the regex pattern to match Python code blocks in Markdown
        pattern = r'```\s*([^`]+)```'
        
        # Use re.findall to extract all matches of the pattern
        matches = re.findall(pattern, s, re.DOTALL)
        
        # Join the matches into a single string
        answer_text = '\n'.join(matches)
        return answer_text

    @staticmethod
    def parser(gpt_raw:str) -> "LeftOrRight":
        """
        @GPT4-doc-begin
            gpt_raw: A block of text that has a single word in a text block, indicated by ```.
                    The word should be either "right" or "left".
                    For example,
                    The question that you've provided asks me to choose the image (left or right) that best aligns with the user text prompt.
                    Though the sample on the left is more realistic, the sample on the right is better aligned with the user prompt.
                    ```
                    right
                    ```
        @GPT4-doc-end
        """
    
        left_or_right = LeftOrRight.remove_answer_text(gpt_raw)
        return LeftOrRight(left_or_right, gpt_raw=gpt_raw) 

    def __str__(self):
        return self.data

class StarredList(ParsedAnswer):
    def __init__(self, list_items:List[str], gpt_raw:str=None):
        self.list_items = list_items 
        self.raw = gpt_raw 

    @staticmethod
    def parse_bullet_points(input_string):
        lines = input_string.split("\n")
        bullet_points = []
        current_bullet_point = ""

        for line in lines:
            if line.startswith("* "):
                if current_bullet_point:
                    bullet_points.append(current_bullet_point.strip())
                    current_bullet_point = ""
                current_bullet_point += line[2:] + " "
            elif current_bullet_point:
                current_bullet_point += line + " "

        if current_bullet_point:
            bullet_points.append(current_bullet_point.strip())

        return bullet_points

    @staticmethod     
    def parser(gpt_raw:str) -> "StarredList":
        """
        @GPT4-doc-begin
            gpt_raw: A new-line separated bulletpoint list that follows the following format:
                
                 Example:
                 * first item
                 * second item
                 ...etc
        @GPT4-doc-end
        """
        list_items = StarredList.parse_bullet_points(gpt_raw)
        return StarredList(list_items=list_items, gpt_raw=gpt_raw)

class PythonExecutableDiffAnswer(ParsedAnswer):
    """ Code (python) difference
    """

    def __init__(self, code_before, code_after, gpt_raw:str=None):
        self.code_from = code_before
        self.code_to = code_after 
        self.raw = gpt_raw

    @staticmethod
    def parser(gpt_raw:str)  -> "PythonExecutableDiffAnswer":
        """ 
        @GPT4-doc-begin
            gpt_raw: A desired code change to a single line, 
                     indicated by "Before:" and "After:" labels,
                     followed by python code blocks that indicate which line should be
                     changed, and to what.
                     Example:
                     
                     Before:
                     ```python
                     a = 1
                     ```
                     After:
                     ```python
                     a = 2 
                     ```
        @GPT4-doc-end 
        """
            
        before_after = gpt_raw.split("After")
        before_text = before_after[0]
        assert "Before" in before_text 
        after_text = before_after[-1]
         
        try: 
            before_string = PythonExecutableAnswer.remove_markdown_code(before_text)
            after_string = PythonExecutableAnswer.remove_markdown_code(after_text)
        except:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(gpt_raw)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            raise GPTOutputParseException(f"Invalid input passsed into parsing function. The following could not be parsed:\n{gpt_raw}")
        return PythonExecutableDiffAnswer(code_before=before_string, code_after=after_string, gpt_raw=gpt_raw)

    def __str__(self):
        return str(self.code)


class PythonExecutableAnswer(ParsedAnswer):
    """ Code (Python)
    """
    def __init__(self, code, gpt_raw:str=None):
        self.code = code 
        self.raw = gpt_raw

    @staticmethod
    def remove_markdown_code(s:str):

        if "```python" not in s:
            raise GPTOutputParseException("No ```python found in the input.")
        chunk1 = s.split("```python")[-1]
        if "```"  not in chunk1:
            raise GPTOutputParseException("No ``` found  to close the code block.")
        python_code = chunk1.split("```")[0]
        return python_code

    @staticmethod
    def parser(gpt_raw:str) -> "PythonExecutableAnswer":
        """ 
        @GPT4-doc-begin
            gpt_raw: string that contains a block of code to be executed in python.
                     you should add comments to explain what you are doing.
                     For instance,
                     We will create a variable, a, and increment it.
                     ```python
                     # create a, and give it an initial value
                     a = 1
                     # increment it
                     a += 1
                     ```
        @GPT4-doc-end 
        """
        
        code_string = PythonExecutableAnswer.remove_markdown_code(gpt_raw)     
        return PythonExecutableAnswer(code=code_string, gpt_raw=gpt_raw)

    def __str__(self):
        return str(self.code)


class TextAnswer(ParsedAnswer):
    """ Text, with minimal parsing
    """
    def __init__(self, data, gpt_raw:str=None): 
        self.data = data
        self.raw = gpt_raw

    @staticmethod 
    def parser(gpt_raw:str):
        data = gpt_raw
        return TextAnswer(data, gpt_raw=gpt_raw)

    def __str__ (self):
        return str(self.data)

# Pass
class YesNoWhy(ParsedAnswer):
    """ Yes/No, but with a reason in the end, and/or suggestions.
    """

    def __init__(self, final_answer, reason_or_suggestions, gpt_raw:str=None):
        self.final_answer = final_answer 
        self.reason_or_suggestions = reason_or_suggestions
        self.raw = gpt_raw
        
    @staticmethod
    def parser(gpt_raw:str):
        """
        @GPT4-doc-begin
        Args:
            gpt_raw: 
                string that contains the tag "[#reason]", where you include the reasoning for
                the final decision (yes/no), followed by the tag "[#finalanswer]", followed by the 
                final answer (yes/no).

                For example,
                    [#reason]
                    your reasoning for why you will answer yes should be put here.

                    [#finalanswer]
                    yes.
                
        @GPT4-doc-end
        """

        if "[#reason]" not in gpt_raw:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(gpt_raw)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")        
            raise GPTOutputParseException(f"{gpt_raw} should have [#reason] tag")
        if "[#finalanswer]" not in gpt_raw:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(gpt_raw)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            raise GPTOutputParseException(f"{gpt_raw} should have [#finalanswer] tag")
        
        gpt_raw_reasoning = "".join(gpt_raw.split("[#reason]")[1:]).split("[#finalanswer]")[0]
        gpt_raw_decision = "".join(gpt_raw.split("[#finalanswer]")[1:])
        
        return YesNoWhy(
                final_answer=str(YesNo.parser(gpt_raw_decision)),
                reason_or_suggestions=str(TextAnswer.parser(gpt_raw_reasoning)), 
                gpt_raw=gpt_raw
        )

    def success(self):
        return self.final_answer== "yes"

    def __str__(self):
        return f"{self.final_answer}: {self.reason_or_suggestions}"
        

class YesNo(ParsedAnswer):
    """ Yes/No
    """ 
    def __init__(self, data, gpt_raw:str=None):
        self.data = data
        self.raw = gpt_raw

    @staticmethod 
    def remove_punctuation(s):
        return re.sub(r'[^\w\s]', '', s)  
        
    @staticmethod
    def parser(gpt_raw:str):
        """
        @GPT4-doc-begin
        Args:
            gpt_raw: string that contains yes/no answers only, and no other words.
            For example,
                yes.
        @GPT4-doc-end
        """
        yesorno = YesNo.remove_punctuation( gpt_raw.lower().strip()) 
        if yesorno== "yes": 
            data = "yes"
        elif yesorno == "no":
            data = "no"
        else:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(gpt_raw)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            raise GPTOutputParseException(f"{yesorno} cannot be parsed to yes/no")
        return YesNo(data, gpt_raw=gpt_raw)

    def success(self):
        """ used in the context of judging the completion of a task.
        """
        return self.data == "yes"
            
    def __str__(self) :
        return self.data


class Number(ParsedAnswer):
    """ Yes/No
    """ 
    def __init__(self, data, gpt_raw:str=None):
        self.data = data
        self.raw = gpt_raw

    @staticmethod 
    def remove_punctuation(s):
        return re.sub(r'[^\w\s]', '', s)  
        
    @staticmethod
    def parser(gpt_raw:str) -> "ReadSign":
        """
        @GPT4-doc-begin
            gpt_raw: string that contains just a simple number.
            
                For example,
                
                90

        @GPT4-doc-end
        """

        gpt_out = gpt_raw.strip().strip('.').strip(',').lower()

        if not gpt_out.isdigit():
            raise GPTOutputParseException("output should only contain a number!")

        return Number(gpt_out, gpt_raw=gpt_raw)

    def success(self):
        """ used in the context of judging the completion of a task.
        """
        return self.data == "yes"
            
    def __str__(self) :
        return self.data