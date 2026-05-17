from abc import ABC,abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelResponse:
    model_name: str
    answer: str
    latency_ms:float
    tokens_used: int
    error:Optional[str]=None

    @property
    def success(self)-> bool:
        return self.error is None
    
class BaseModel(ABC):
    def __init__(self,model_id:str,temperature:float=0.7,max_tokens:int=1024):
        self.model_id=model_id
        self.temperature=temperature
        self.max_tokens=max_tokens

    @abstractmethod
    async def generate(self,prompt:str,system_prompt: str="") -> ModelResponse:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model_id={self.model_id})"