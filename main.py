"""
RAG Pipeline - TCC
Comparison of LLM Models for Data Retrieval from Static Pages.

Usage:
    python main.py                                  # Default config
    python main.py --model "mistral:latest"         # Override model
    python main.py --url "https://..." --input X    # Override URL and input
    python main.py --skip-ragas                     # Skip evaluation
    python main.py --backend huggingface            # Use HuggingFace
    python main.py --few-shot                       # Enable few-shot
    python main.py --config config/custom.yaml      # Custom config
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline for LLM comparison on static pages"
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--backend", choices=["ollama", "huggingface"],
        help="LLM backend to use (overrides config)",
    )
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--url", help="Override document URL")
    parser.add_argument("--input", help="Override input CSV path")
    parser.add_argument("--output", help="Override output CSV path")
    parser.add_argument(
        "--few-shot", action="store_true",
        help="Enable few-shot prompting",
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAGAS evaluation step",
    )
    args = parser.parse_args()

    # Lazy import to keep --help fast
    import yaml
    from src.llm import build_llm
    from src.document import load_and_split
    from src.retriever import build_retriever
    from src.chain import build_qa_chain
    from src.evaluate import run_ragas
    from src.utils import process_questions, unload_model

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.backend:
        cfg["backend"] = args.backend
    if args.model:
        backend = cfg.get("backend", "ollama")
        if backend == "ollama":
            cfg["ollama"]["model"] = args.model
        else:
            cfg["huggingface"]["model_name"] = args.model
    if args.url:
        cfg["document"]["url"] = args.url
    if args.input:
        cfg["io"]["input_csv"] = args.input
    if args.output:
        cfg["io"]["output_csv"] = args.output
    if args.few_shot:
        cfg["chain"]["few_shot"] = True

    # ---- Pipeline ----
    llm, model_name = build_llm(cfg)
    docs = load_and_split(cfg)
    retriever, embeddings = build_retriever(docs, cfg)
    qa_chain = build_qa_chain(llm, retriever, cfg)
    questions, answers, contexts = process_questions(qa_chain, cfg)

    # Unload Ollama model to free memory before RAGAS evaluation
    if cfg.get("backend", "ollama") == "ollama":
        unload_model(model_name)

    if not args.skip_ragas:
        run_ragas(questions, answers, contexts, embeddings, cfg)


if __name__ == "__main__":
    main()
