import time 
import os
from huggingface_hub import AsyncInferenceClient
from Models.base import BaseModel,ModelResponse


class HuggingFaceModel(BaseModel):
    def __init__(self,model_id:str="Qwen/Qwen2.5-72B-Instruct",temperature:float=0.7,max_tokens:int=1024):
        super().__init__(model_id,temperature,max_tokens)
        self.api_key=os.getenv("HUGGINGFACE_API_KEY")
        self.client=AsyncInferenceClient(token=self.api_key, timeout=120)
        self.name="Qwen-72B"

    async def generate(self,prompt:str,system_prompt:str="") -> ModelResponse:
        star=time.time()
        try:
            messages=[]
            if system_prompt:
                messages.append({"role":"system","content":system_prompt})
            messages.append({"role":"user","content":prompt})
            
            response = await self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Robust validation for API edge cases
            if not response:
                raise ValueError("Received empty response from HuggingFace API")
                
            if hasattr(response, "choices") and response.choices:
                answer = response.choices[0].message.content
                tokens = getattr(response.usage, "total_tokens", len(answer.split()))
            elif isinstance(response, dict) and "choices" in response and response["choices"]:
                answer = response["choices"][0]["message"]["content"]
                usage = response.get("usage", {})
                tokens = usage.get("total_tokens", len(answer.split()))
            else:
                raise ValueError(f"Unexpected response format: {response}")
                
            latency=(time.time()-star)*1000
            
            return ModelResponse(
                model_name=self.name,
                answer=answer,
                latency_ms=round(latency,2),
                tokens_used=tokens,
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                answer="",
                latency_ms=round((time.time()-star)*1000,2),
                tokens_used=0,
                error=str(e)
            )