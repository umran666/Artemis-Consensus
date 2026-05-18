from typing import Dict, List
import os
from pipeline.ensemble import EnsemblePipeline, EnsembleResult
from datasets import load_dataset
import pandas as pd

def load_truthfulqa(sample_size: int = 100) -> List[Dict]:
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    samples = list(dataset.select(range(min(sample_size, len(dataset)))))
    return [
        {
            "question": sample["question"],
            "correct_answers": sample["mc1_targets"]["choices"],
        }
        for sample in samples
    ]


def load_gsm8k(sample_size: int = 100) -> List[Dict]:
    dataset = load_dataset("gsm8k", "main", split="train")
    samples = list(dataset.select(range(min(sample_size, len(dataset)))))
    return [
        {
            "question": sample["question"],
            "answer": sample["answer"],
        }
        for sample in samples
    ]


def check_answer_truthfulqa(final_answer: str, correct_answers: List[str]) -> bool:
    """Simple keyword matching. Replace with embedding similarity for better accuracy."""
    answer_lower = final_answer.lower()
    return any(a.lower() in answer_lower for a in correct_answers)


def check_answer_gsm8k(final_answer: str, ground_truth: str) -> bool:
    """Extract final number from answer and compare."""
    import re
    numbers_pred = re.findall(r"[-+]?\d*\.?\d+", final_answer.replace(",", ""))
    numbers_true = re.findall(r"[-+]?\d*\.?\d+", ground_truth.replace(",", ""))
    if not numbers_pred or not numbers_true:
        return False
    return numbers_pred[-1] == numbers_true[-1]


async def run_benchmark(
    pipeline: EnsemblePipeline,
    benchmark: str = "truthfulqa",
    sample_size: int = 50,
    output_dir: str = "data/results",
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    if benchmark == "truthfulqa":
        samples = load_truthfulqa(sample_size)
    elif benchmark == "gsm8k":
        samples = load_gsm8k(sample_size)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    
    results = []
    print(f"\nRunning benchmark '{benchmark}' with {len(samples)} samples...\n")

    for i,sample in enumerate(samples):
        question = sample["question"]
        try:
            result: EnsembleResult = await pipeline.run(question)
            if benchmark == "truthfulqa":
                ensemble_correct = check_answer_truthfulqa(result.final_answer, sample["correct_answers"])
                individual_correct = {
                    r.model_name: check_answer_truthfulqa(r.answer, sample["correct_answers"])
                    for r in result.model_responses if r.success
                }
            else:
                ensemble_correct = check_answer_gsm8k(result.final_answer, sample["answer"])
                individual_correct = {
                    r.model_name: check_answer_gsm8k(r.answer, sample["answer"])
                    for r in result.model_responses if r.success
                }

            row = {
                "question": question,
                "ensemble_correct": ensemble_correct,
                "latency_ms": result.total_latency_ms,
                "tokens_used": result.total_tokens,
                "disagreement": result.disagreement_detected,
                "confidence": result.final_score.weighted_total if result.final_score else 0,
            }
            row.update({f'{k}_correct': v for k, v in individual_correct.items()})
            results.append(row)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")


    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, f"{benchmark}_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nBenchmark completed. Results saved to {output_path}\n")
    print_benchmark_summary(df)
    return df


def print_benchmark_summary(df: pd.DataFrame):
    if df.empty:
        print("\nNo benchmark results were generated.")
        return

    print("\n"+"="*50)
    print("Benchmark Summary:")
    print("="*50)
    correct_cols=[c for c in df.columns if c.endswith("_correct")]
    for col in correct_cols:
        label = col.replace("_correct", "").replace("_", " ").title()
        accuracy = df[col].mean() * 100
        print(f"{label:<30} {accuracy:.1f}%")
    print("=" * 50)
