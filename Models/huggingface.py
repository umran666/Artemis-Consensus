import json
import os
import time
from huggingface_hub import AsyncInferenceClient
from Models.base import BaseModel,ModelResponse


class HuggingFaceModel(BaseModel):
    def __init__(self,model_id:str="Qwen/Qwen2.5-72B-Instruct",temperature:float=0.7,max_tokens:int=8192):
        super().__init__(model_id,temperature,max_tokens)
        self.api_key=os.getenv("HUGGINGFACE_API_KEY")
        self.client=AsyncInferenceClient(token=self.api_key, timeout=120)
        self.name="Qwen-72B"

    def _extract_text(self, content) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                else:
                    text = getattr(item, "text", None) or getattr(item, "content", None)

                if text:
                    parts.append(str(text).strip())

            return "\n".join(part for part in parts if part).strip()

        return ""

    def _response_tokens(self, response, answer: str) -> int:
        usage = getattr(response, "usage", None)
        if usage is not None:
            total_tokens = getattr(usage, "total_tokens", None)
            if total_tokens is not None:
                return total_tokens

        if isinstance(response, dict):
            usage = response.get("usage")
            if isinstance(usage, dict) and usage.get("total_tokens") is not None:
                return usage["total_tokens"]

        return len(answer.split())

    def _parse_chat_completion(self, response) -> tuple[str, int]:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")

        if not choices:
            raise ValueError(f"No choices found in HuggingFace chat response: {response!r}")

        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
        else:
            message = getattr(first_choice, "message", None)

        if message is None:
            raise ValueError(f"Missing message in HuggingFace response choice: {first_choice!r}")

        if isinstance(message, dict):
            content = message.get("content")
            tool_calls = message.get("tool_calls")
        else:
            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None)

        answer = self._extract_text(content)

        if not answer and tool_calls:
            answer = json.dumps(tool_calls, default=str, ensure_ascii=True)

        if not answer:
            raise ValueError(f"No usable content found in HuggingFace response message: {message!r}")

        return answer, self._response_tokens(response, answer)

    async def _generate_fallback_text(self, prompt: str, system_prompt: str) -> tuple[str, int]:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = await self.client.text_generation(
            prompt=full_prompt,
            model=self.model_id,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            return_full_text=False,
            details=True,
        )

        if isinstance(response, str):
            answer = response.strip()
            tokens = len(answer.split())
        elif isinstance(response, dict):
            answer = str(response.get("generated_text", "")).strip()
            details = response.get("details") or {}
            tokens = details.get("generated_tokens", len(answer.split()))
        else:
            answer = str(getattr(response, "generated_text", "")).strip()
            details = getattr(response, "details", None)
            tokens = getattr(details, "generated_tokens", len(answer.split())) if details else len(answer.split())

        if not answer:
            raise ValueError(f"No usable text returned by HuggingFace text_generation: {response!r}")

        return answer, tokens

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

            answer, tokens = self._parse_chat_completion(response)
        except Exception as chat_error:
            try:
                answer, tokens = await self._generate_fallback_text(prompt, system_prompt)
            except Exception as fallback_error:
                return ModelResponse(
                    model_name=self.name,
                    answer="",
                    latency_ms=round((time.time()-star)*1000,2),
                    tokens_used=0,
                    error=f"chat_completion failed: {chat_error}; text_generation fallback failed: {fallback_error}"
                )

        try:
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
