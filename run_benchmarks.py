import asyncio
import argparse
from dotenv import load_dotenv
load_dotenv()

from Models.groq import GroqModel
from Models.gemini import GeminiModel
from Models.huggingface import HuggingFaceModel
from pipeline.ensemble import EnsemblePipeline
from evaluation.benchmark import run_benchmark



async def main(benchmark_name: str, samples: int):
    models = [GroqModel(),GeminiModel(),HuggingFaceModel()]
    pipeline = EnsemblePipeline(models)
    await run_benchmark(pipeline,benchmark=benchmark_name,sample_size=samples)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Run benchmarks on the ensemble pipeline.")

    parser.add_argument("--benchmark_name",choices=["gsm8k","truthfulqa"],default="truthfulqa")
    parser.add_argument("--samples",type=int,default=100)
    args=parser.parse_args()
    asyncio.run(main(args.benchmark_name,args.samples))
