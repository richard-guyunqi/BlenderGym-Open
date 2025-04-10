from loguru import logger
from .exceptions import UnreadableGPTDocumentation

def docs_for_GPT4(func, docstring_tag="@GPT4-doc-") -> str:
    docstring = func.__doc__
    if docstring is None:
        raise UnreadableGPTDocumentation(f"{func} does not have a doc string.")

    begin = docstring_tag + "begin"
    end = docstring_tag + "end"
    
    if not (begin in docstring and end in docstring):
        raise UnreadableGPTDocumentation(f"{func} does not have {begin} and {end} to indicate the beginning and end of GPT-readable documentation")

    gpt4docstring = (''.join(docstring.split(begin)[1:])).split(end)[0]
    return gpt4docstring
    
class URL(object):
    def __init__(self, url):
        self.url = url
        
    def __str__(self):
        return self.url 

