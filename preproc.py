"""
preprocessing and tokenizing
"""
import os 
import json 
from abc import abstractmethod, ABC
from pathlib import Path 


START_SEQ = '<s>'
END_SEQ = '</s>'


class BasePreprocStrategy(ABC): 
    @abstractmethod
    def __call__(self) -> list[str]:
        """
        implement preprocessing behaviour here.
        the expected output is a list of strings that
        represent semantically meaningful, self-contained
        units of text.
        """
        pass 
    
    def join(self, texts: list[str], delim: str = '\n') -> list[str]: 
        return delim.join([f'{START_SEQ} {text} {END_SEQ}' for text in texts])

    def export(self, path: Path = './out.txt'):
        texts  = self.__call__() 
        res = self.join(texts) 
        with open(path, 'w') as f: 
            f.write(res)


class TgPreporcStrategy(BasePreprocStrategy):
    def __init__(self, path: Path, id_: str): 
        with open(path, 'r') as f:
            self.cnv = json.load(f)
        self.target = id_
        self.lower = True

    def __call__(self):
        msgs = [] 
        for msg in self.cnv['messages']:
            
            if msg.get('from_id') == self.target and isinstance(msg.get('text'), str):
                text = msg['text']
                if self.lower:
                    text = text.lower() 

                # empty str check 
                if text.strip():
                    msgs.append(text)

        return msgs 

if __name__ == '__main__': 
    path = './data/ChatExport_2026-06-20/result.json'
    preproc = TgPreporcStrategy(
        path,
        id_=os.environ.get('USER_ID')
    ).export()
