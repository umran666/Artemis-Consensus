import time 
import os
import httpx
from Models.base import BaseModel,ModelResponse


class HuggingFaceModel(BaseModel):
    def __init__(self,model_id:str="mistralai/Mistral-7B-Instruct-v0.2",temperature:float=0.7,max_tokens:int=1024):
        super().__init__(model_id,temperature,max_tokens)
        self.api_key=os.getenv("Huggingface_api_key")
        self.api_url=f"https://api-inference.huggingface.co/models/{model_id}"
        self.name="Mistral-7B"

    async def generate(self,prompt:str,system_prompt:str="") -> ModelResponse:
        star=time.time()
        try:
            full_prompt=f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]" if system_prompt else f"<s>[INST] {prompt} [/INST]"
            headers={"Authorization":f"Bearer{self.api_key}"}
            payload={
                "inputs":full_prompt,
                "parameters":{
                    "temperature":self.temperature,
                    "max_new_tokens":self.max_tokens,
                    "return_full_text":False,
                },

            }
            async with httpx.AsyncClient(timeout=30) as client:
                response=await(client.post(self.api_url,headers=headers,json=payload))
                response.raise_for_status()
                data=response.json()
            latency=(time.time()-star)*1000
            answer=data[0]["generated_text"] if isinstance(data,list) else data.get("generated_text","")
            return ModelResponse(
                model_name=self.name,
                answer=answer,
                latency_ms=round(latency,2),
                tokens_used=len(answer.split()),
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                answer="",
                latency_ms=round((time.time()-star)*1000,2),
                tokens_used=0,
                error=str(e)
            )