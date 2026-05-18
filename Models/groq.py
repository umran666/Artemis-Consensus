import os
import time
from groq import AsyncGroq
from Models.base import BaseModel,ModelResponse

class GroqModel(BaseModel):
    def __init__(self,model_id:str="llama-3.3-70b-versatile",temperature:float=0.7,max_tokens:int=8192):
        super().__init__(model_id,temperature,max_tokens)
        self.client=AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.name="Llama 3(Groq)"

    async def generate(self,prompt:str,system_prompt:str="") -> ModelResponse:
        start=time.time()
        try:
            messages=[]
            if system_prompt:
                messages.append({"role":"system","content":system_prompt})
            messages.append({"role":"user","content":prompt})
            response=await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            latency=(time.time()-start)*1000
            return ModelResponse(
                model_name=self.name,
                answer=response.choices[0].message.content,
                latency_ms=round(latency,2),
                tokens_used=response.usage.total_tokens,

            )
        
        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                answer="",
                latency_ms=round((time.time()-start)*1000,2),
                tokens_used=0,
                error=str(e),

            )