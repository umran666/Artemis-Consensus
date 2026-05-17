import asyncio
from typing import List, Dict, Any, Optional

from Models.base import BaseModel, ModelResponse
from pipeline.scorer import Score, parse_critique_response


CRITIQUE_SYSTEM_PROMPT = """You are a strict evaluator. You will be given a question and an answer from another AI model.

Evaluate the answer on these 5 dimensions and give a score from 0 to 10 for each:

- factuality: Are the claims accurate and well-supported?
- confidence: Is the confidence level appropriate, not overconfident or underconfident?
- completeness: Does it fully answer the question without missing key points?
- consistency: Is the answer internally consistent with no contradictions?
- reasoning: Is the logic sound and well-structured?

Respond ONLY with valid JSON in this exact format:
{
  "factuality": X,
  "confidence": X,
  "completeness": X,
  "consistency": X,
  "reasoning": X,
  "feedback": "One sentence explaining the biggest weakness"
}
"""


def get_model_name(model: BaseModel) -> str:
    """
    Safely get model name from BaseModel.

    Supports:
    - model.name
    - model.model_name
    - fallback to class name
    """

    if hasattr(model, "name") and model.name:
        return model.name

    if hasattr(model, "model_name") and model.model_name:
        return model.model_name

    return model.__class__.__name__


async def critique_single(
    critic: BaseModel,
    question: str,
    answer_to_critique: str,
    answer_model_name: str,
) -> Dict[str, Any]:
    """
    One model critiques one answer.
    """

    critic_name = get_model_name(critic)

    prompt = f"""Question:
{question}

Answer from {answer_model_name}:
{answer_to_critique}

Now evaluate this answer using the required JSON scoring format."""

    response: ModelResponse = await critic.generate(
        prompt,
        system_prompt=CRITIQUE_SYSTEM_PROMPT,
    )

    if not response or not response.answer:
        raise RuntimeError(
            f"Critic {critic_name} returned empty critique for {answer_model_name}"
        )

    score, feedback = parse_critique_response(response.answer)

    return {
        "critic": critic_name,
        "target_model": answer_model_name,
        "score": score,
        "feedback": feedback,
        "latency_ms": response.latency_ms or 0,
        "tokens_used": response.tokens_used or 0,
    }


async def run_cross_critique(
    models: List[BaseModel],
    question: str,
    responses: List[ModelResponse],
) -> Dict[str, Any]:
    """
    Each model critiques the other models' answers.

    Example:
    - Gemini critiques Groq and Mistral
    - Groq critiques Gemini and Mistral
    - Mistral critiques Gemini and Groq

    Returns:
    {
        "scores": {
            "model_name": Score(...)
        },
        "feedback": {
            "model_name": ["critic: feedback", ...]
        },
        "raw_results": [...]
    }
    """

    tasks = []

    valid_responses = [
        response
        for response in responses
        if response.success and response.answer and response.answer.strip()
    ]

    if len(valid_responses) < 1:
        return {
            "scores": {},
            "feedback": {},
            "raw_results": [],
        }

    for critic in models:
        critic_name = get_model_name(critic)

        for response in valid_responses:
            if critic_name == response.model_name:
                continue

            tasks.append(
                critique_single(
                    critic=critic,
                    question=question,
                    answer_to_critique=response.answer,
                    answer_model_name=response.model_name,
                )
            )

    if not tasks:
        return {
            "scores": {},
            "feedback": {},
            "raw_results": [],
        }

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    critique_map: Dict[str, List[Score]] = {}
    critique_feedback: Dict[str, List[str]] = {}
    raw_results: List[Dict[str, Any]] = []

    for result in results:
        if isinstance(result, Exception):
            print(f"[Critique Error] {result}")
            continue

        target_model = result["target_model"]

        if target_model not in critique_map:
            critique_map[target_model] = []
            critique_feedback[target_model] = []

        critique_map[target_model].append(result["score"])
        critique_feedback[target_model].append(
            f"{result['critic']}: {result['feedback']}"
        )

        raw_results.append(result)

    average_scores: Dict[str, Score] = {}

    for model_name, scores in critique_map.items():
        if not scores:
            continue

        average_scores[model_name] = Score(
            factuality=sum(score.factuality for score in scores) / len(scores),
            confidence=sum(score.confidence for score in scores) / len(scores),
            completeness=sum(score.completeness for score in scores) / len(scores),
            consistency=sum(score.consistency for score in scores) / len(scores),
            reasoning=sum(score.reasoning for score in scores) / len(scores),
        )

    return {
        "scores": average_scores,
        "feedback": critique_feedback,
        "raw_results": raw_results,
    }


async def critique_responses(
    critic_model: Optional[BaseModel] = None,
    question: Optional[str] = None,
    responses: Optional[List[ModelResponse]] = None,
    models: Optional[List[BaseModel]] = None,
) -> Dict[str, Any]:
    """
    Compatibility wrapper used by EnsemblePipeline.

    Supports two modes:

    1. Single critic mode:
       critique_responses(
           critic_model=some_model,
           question=question,
           responses=responses
       )

    2. Cross-critique mode:
       critique_responses(
           models=models,
           question=question,
           responses=responses
       )
    """

    if question is None:
        raise ValueError("question is required.")

    if responses is None:
        raise ValueError("responses is required.")

    valid_responses = [
        response
        for response in responses
        if response.success and response.answer and response.answer.strip()
    ]

    if not valid_responses:
        return {
            "scores": {},
            "feedback": {},
            "raw_results": [],
        }

    # Full cross-critique mode
    if models:
        return await run_cross_critique(
            models=models,
            question=question,
            responses=responses,
        )

    # Single critic mode
    if critic_model is None:
        raise ValueError("Either critic_model or models must be provided.")

    tasks = []

    for response in valid_responses:
        tasks.append(
            critique_single(
                critic=critic_model,
                question=question,
                answer_to_critique=response.answer,
                answer_model_name=response.model_name,
            )
        )

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    scores: Dict[str, Score] = {}
    feedback: Dict[str, List[str]] = {}
    raw_results: List[Dict[str, Any]] = []

    for result in results:
        if isinstance(result, Exception):
            print(f"[Critique Error] {result}")
            continue

        target_model = result["target_model"]

        scores[target_model] = result["score"]
        feedback[target_model] = [
            f"{result['critic']}: {result['feedback']}"
        ]

        raw_results.append(result)

    return {
        "scores": scores,
        "feedback": feedback,
        "raw_results": raw_results,
    }