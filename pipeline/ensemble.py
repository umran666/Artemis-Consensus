import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from Models.base import BaseModel, ModelResponse
from pipeline.scorer import Score
from pipeline.synthesis import synthesize
from pipeline.critique import critique_responses


@dataclass
class EnsembleResult:
    question: str
    model_responses: List[ModelResponse]
    critique_scores: Dict[str, Score]
    critique_feedback: Dict[str, List[str]]
    final_answer: str
    final_score: Optional[Score]
    disagreement_detected: bool
    total_latency_ms: float
    total_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "disagreement_detected": self.disagreement_detected,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "confidence_label": (
                self.final_score.confidence_label
                if self.final_score
                else "Unknown"
            ),
            "model_responses": [
                {
                    "model": r.model_name,
                    "answer": r.answer,
                    "latency_ms": r.latency_ms,
                    "tokens": r.tokens_used,
                    "error": r.error,
                }
                for r in self.model_responses
            ],
            "critique_scores": {
                model_name: score.to_dict()
                for model_name, score in self.critique_scores.items()
            },
            "critique_feedback": self.critique_feedback,
        }


def detect_disagreement(
    responses: List[ModelResponse],
    threshold: float = 0.6,
) -> bool:
    """
    Detect disagreement between model answers using TF-IDF cosine similarity.

    If average similarity is lower than threshold, disagreement=True.
    """

    valid_responses = [
        response
        for response in responses
        if response.success and response.answer and response.answer.strip()
    ]

    if len(valid_responses) < 2:
        return False

    texts = [response.answer for response in valid_responses]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)

        similarity_matrix = cosine_similarity(tfidf_matrix)

        n = len(valid_responses)

        pairwise_scores = [
            similarity_matrix[i][j]
            for i in range(n)
            for j in range(i + 1, n)
        ]

        if not pairwise_scores:
            return False

        average_similarity = sum(pairwise_scores) / len(pairwise_scores)

        return average_similarity < threshold

    except Exception as e:
        print(f"[Disagreement Detection Error] {e}")
        return True


def average_scores(scores: Dict[str, Score]) -> Optional[Score]:
    """
    Create one average final score from all critique scores.
    """

    if not scores:
        return None

    score_values = list(scores.values())

    return Score(
        factuality=sum(score.factuality for score in score_values) / len(score_values),
        confidence=sum(score.confidence for score in score_values) / len(score_values),
        completeness=sum(score.completeness for score in score_values) / len(score_values),
        consistency=sum(score.consistency for score in score_values) / len(score_values),
        reasoning=sum(score.reasoning for score in score_values) / len(score_values),
    )


async def run_cross_critique(
    models: List[BaseModel],
    question: str,
    responses: List[ModelResponse],
) -> Dict[str, Any]:
    """
    Runs critique step.

    Expected critique_responses return format:

    {
        "scores": {
            "model_name": Score(...)
        },
        "feedback": {
            "model_name": ["feedback 1", "feedback 2"]
        }
    }
    """

    critique_result = await critique_responses(
        models=models,
        question=question,
        responses=responses,
    )

    if critique_result is None:
        return {
            "scores": {},
            "feedback": {},
        }

    if not isinstance(critique_result, dict):
        raise TypeError(
            "critique_responses must return a dict with keys: scores, feedback"
        )

    return {
        "scores": critique_result.get("scores", {}),
        "feedback": critique_result.get("feedback", {}),
    }


class EnsemblePipeline:
    def __init__(
        self,
        models: List[BaseModel],
        disagreement_threshold: float = 0.6,
        always_critique: bool = True,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.disagreement_threshold = disagreement_threshold
        self.always_critique = always_critique

    async def run(self, question: str) -> EnsembleResult:
        start_time = time.time()
        tasks = [
            model.generate(question)
            for model in self.models
        ]

        responses: List[ModelResponse] = await asyncio.gather(
            *tasks,
            return_exceptions=False,
        )

        valid_responses = [
            response
            for response in responses
            if response.success and response.answer and response.answer.strip()
        ]

        if not valid_responses:
            raise RuntimeError("All models failed to respond.")

        disagreement_detected = detect_disagreement(
            responses=responses,
            threshold=self.disagreement_threshold,
        )

        critique_scores: Dict[str, Score] = {}
        critique_feedback: Dict[str, List[str]] = {}

        should_critique = (
            len(valid_responses) >= 2
            and (self.always_critique or disagreement_detected)
        )

        critic_model = self.models[0]

        if should_critique:
            critique_result = await run_cross_critique(
                models=self.models,
                question=question,
                responses=responses,
            )

            critique_scores = critique_result.get("scores", {})
            critique_feedback = critique_result.get("feedback", {})

        synthesizer = self.models[0]

        final_response: ModelResponse = await synthesize(
            synthesizer=synthesizer,
            question=question,
            responses=responses,
            scores=critique_scores,
            feedback=critique_feedback,
        )

        if not final_response or not final_response.answer:
            error_msg = final_response.error if final_response else "No response"
            raise RuntimeError(
                f"Synthesis failed to produce a final answer. Error: {error_msg}"
            )

        final_score = average_scores(critique_scores)

        total_latency_ms = round((time.time() - start_time) * 1000, 2)

        total_tokens = sum(
            response.tokens_used or 0
            for response in responses
        )

        total_tokens += final_response.tokens_used or 0

        return EnsembleResult(
            question=question,
            model_responses=responses,
            critique_scores=critique_scores,
            critique_feedback=critique_feedback,
            final_answer=final_response.answer,
            final_score=final_score,
            disagreement_detected=disagreement_detected,
            total_latency_ms=total_latency_ms,
            total_tokens=total_tokens,
        )