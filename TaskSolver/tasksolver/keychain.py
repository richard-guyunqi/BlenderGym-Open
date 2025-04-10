import os
from typing import Union, Dict

class KeyChain(object):
    def __init__(self,keys:Union[None, Dict[str, str]]=None):
        if keys is None:
            self.keys = {}
        else:
            assert isinstance(keys, dict), "Keys should be dict."
            
         
    def add_key(self, service:str, key:str):
        if os.path.exists(key): # it's a file
            with open(key, "r") as f:
                key = f.readline().strip()
        self.keys.update({service: key}) 
        return self

    def get_key(self, service:str ):
        if service not in self.keys:
            raise ValueError(f"No keys associated with '{service}'") 

        return self.keys[service]

    def __getitem__(self, service:str):
        return self.get_key(service)
     