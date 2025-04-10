"""
The atomic unit of a memory is an event.
A collection of events form an EventCollection.
"""

import json
from typing import List, Tuple, Union
from .common import Question, ParsedAnswer
from bson import ObjectId
import datetime
import time

# TODO consider how to embed images in json files too.
def file_location_type(path:str) -> str:
    """ given a type, it returns a string indicating whether this is `local`, or `url`
    """
    if path.startswith('http://') or path.startswith('https://'):
        return 'url'
    return 'local' 

def read_event_file(file_location:str) -> dict:
    """ Event files loaded as json
    """
    with open(file_location, 'r') as f:
        values = json.load(f)
    return values

def write_event_file(file_location:str, values: dict): 
    """ Event files stored as json.
    """
    prepare_dir_of(file_location)
    if file_location_type(file_location) == 'local':
        with open(file_location, 'w') as f:
            json.dump(values,f)
    else:
        raise NotImplementedError(f'Uncaught file location type {fl_type}')
    return file_location


class EventCollection(object):
    """ Class used for data processing from event files
    """
    def __init__(self):
        self.id = str(ObjectId())
        self.events = []

    def load_from_session(self, event_dir:str, session_token:str) -> "EventCollection":
        if file_location_type (event_dir) == 'local':
            assert os.path.exists(os.path.join(event_dir, session_token)), f'{os.path.join(event_dir, session_token)} does not exist'
            
            for event_file in os.listdir(os.path.join(event_dir, session_token)):
                # try:
                new_event = Event().load_from_event_file(os.path.join(event_dir, session_token, event_file))
                # except:
                #     logger.info(f'Error loading from {os.path.join(event_dir, session_token, event_file)}')
                #     continue
                self.add_event(new_event)
        else:
            raise NotImplementedError(f'Uncaught file location {file_location_type(event_dir)}')

        self.time_sorted()
        return self

    def add_event(self, event:"Event") -> "EventCollection":
        assert isinstance(event, Event), f"invalid type {type(event)} for EventCollection"
        self.events.append(event)
        return self

    def time_sorted(self) -> "EventCollection":
        self.events = sorted(self.events)
        return self

    def __str__(self) -> str:
        printout = ""
        for ev in sorted(self.events):
            printout += str(ev)+'\n'
        return printout

    def  __len__(self) -> int:
        return len(self.events)

    def filter_to(self, label:Union[str, list[str]]):
        if isinstance(label, str):
            label = [label]
        
        permissible_classes = []
        for l in label: 
            assert l in TYPE2CLASS, f"unknown type {label}"
            permissible_classes.append(TYPE2CLASS[l])
             
        filtered_events = [ev for ev in self.events if isinstance(ev, tuple(permissible_classes))]
        return filtered_events


class Event(object):
    def __init__(self, session_token:str = None):
        self.session_token = session_token
        self.timestamp = str(datetime.datetime.now())
        self.type = 'EVENT'

    # comparison methods for sorting
    def __lt__(self, obj: "Event") -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp < obj.timestamp
    def __gt__(self, obj: "Event") -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp > obj.timestamp
    def __le__(self, obj: "Event") -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp <= obj.timestamp
    def __ge__(self, obj: "Event") -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp >= obj.timestamp
    def __eq__(self, obj: "Event") -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp == obj.timestamp 

    def load_from_event_params(self, **kwargs) -> "Event": 
        for key in kwargs:
            setattr(self, key, kwargs[key])
        return self

    def load_from_event_file(self, filepath: str) -> "Event":
        fields = read_event_file(filepath) 
        return TYPE2CLASS[fields['type']]().load_from_event_params(**fields) 

    def export(self) -> dict:
        return vars(self)

    def save_to_event_file(self, filepath: str) -> str:
        attr_dict = self.export() 
        if hasattr(self, 'latency') and self.latency is None:
            raise ValueError("Forgot to call .tick and .tock?")
        write_event_file(filepath, attr_dict)
        return filepath

    @property
    def description(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return f'{str(self.timestamp)}: [{self.type}]\n{self.description}'


class ThinkEvent(Event):
    def __init__(self, session_token:str = None, qa_sequence:List[Tuple[Question, ParsedAnswer]]=None):
        super().__init__(session_token)
        assert isinstance(qa_sequence, list), "qa_sequence should be least, even if len 1."
        assert all([isinstance(el, tuple) for el in qa_sequence]), "Elements of qa_sequence should be tuple."

        self.qa_sequence = qa_sequence

    @property
    def description(self) -> str:
        st = ""
        for idx, qa in enumerate(qa_sequence):
            q, a = qa
            st += f"#### Question {idx}\n"
            st += str(q)
            st += "\n"
            st += f"#### Answer{idx}\n"
            st += str(a)
            st += "\n"
        return st
        
class ActEvent(Event):
    def __init__(self, session_token:str = None):
        super().__init__(session_token)

    @property
    def description(self) -> str:
        raise NotImplementedError

class ActErrorEvent(Event):
    """ Event of an error in the execution of an action.
    """
    def __init__(self, session_token:str = None,
                 exception:Exception=None):
        super().__init__(session_token)
        self.exception = exception

    @property
    def description(self) -> str:
        raise NotImplementedError

    
class ObserveEvent(Event): 
    def __init__(self, session_token:str = None):
        super().__init__(session_token)
    
    @property
    def description(self) -> str:
        raise NotImplementedError

class EvaluateEvent(Event):
    def __init__(self, session_token:str = None,
                 completion_question:Question = None,
                 completion_eval:ParsedAnswer = None):
        super().__init__(session_token)
        assert isinstance(completion_question, Question) 
        assert isinstance(completion_eval, ParsedAnswer)
        self.completion_question = completion_question
        self.completion_eval = completion_eval
    
    @property
    def description(self) -> str:
        st = "## Evaluation Question\n"
        st += "\n"
        st += str(self.completion_question)
        st += "\n"
        st += "## Evaluation Answer\n"
        st += "\n"
        st += str(self.completion_eval)
        st += "\n"
        return st 

class FeedbackEvent(Event):
    def __init__(self, session_token:str = None,
                 feedback:Union[None, Question]=None):
        super().__init__(session_token)
        assert isinstance(feedback, Question) or feedback is None
        self.feedback = feedback

    @property
    def description(self)  -> str:
        st = "## Reflection:\n"
        st += str(self.feedback)
        st += "\n"
        return st


class InteractEvent(Event):
    def __init__(self, session_token:str = None):
        super().__init__(session_token)

    @property
    def description(self) -> str:
        raise NotImplementedError


TYPE2CLASS = {
'EVENT': Event, 
# TAORI events
'THINK': ThinkEvent, 
'ACT': ActEvent,
'ACTERROR': ActErrorEvent,
'OBSERVE': ObserveEvent,
'EVALUATE': EvaluateEvent,
'FEEDBACK': FeedbackEvent,
'INTERACT': InteractEvent,
}