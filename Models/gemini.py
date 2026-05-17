import time 
import os 
from google import genai
from Models.base import BaseModel,ModelResponse

class GeminiModel(BaseModel):
    def __init__(self,model_id: str="gemini-2.5-flash-lite",temperature:float=0.7,max_tokens:int=1024):
        super().__init__(model_id,temperature,max_tokens)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.name="Gemini Flash"
    async def generate(self,prompt:str,system_prompt:str="") -> ModelResponse:
        start=time.time()
        try:
            full_prompt=f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            latency=(time.time()-start)*1000
            text=response.text
            return ModelResponse(
                model_name=self.name,
                answer=text,
                latency_ms=round(latency,2),
                tokens_used=response.usage_metadata.total_token_count if response.usage_metadata else 0,
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                answer="",
                latency_ms=round((time.time()-start)*1000,2),
                tokens_used=0,
                error=str(e)
            )