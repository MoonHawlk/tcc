"""Utility functions: question processing and model management."""

import time
import subprocess
import pandas as pd


def process_questions(qa_chain, cfg: dict) -> tuple:
    """Read questions from CSV, run them through the QA chain, and save answers.

    Args:
        qa_chain: RetrievalQA chain instance.
        cfg: Full pipeline configuration dict.

    Returns:
        (questions, answers, contexts) tuple for downstream evaluation.
    """
    io_cfg = cfg["io"]

    df = pd.read_csv(io_cfg["input_csv"])
    if "pergunta" not in df.columns:
        raise ValueError("Coluna 'pergunta' nao encontrada no CSV.")

    questions = df["pergunta"].tolist()
    answers = []
    contexts = []

    start = time.time()
    for q in questions:
        print(f"\nPergunta: {q}")
        out = qa_chain.invoke({"query": q})
        answers.append(out["result"])
        contexts.append([d.page_content for d in out["source_documents"]])
        print(f"Resposta: {answers[-1]}")

    elapsed = time.time() - start

    df["resposta"] = answers
    df.to_csv(io_cfg["output_csv"], index=False)
    print(f"\nRespostas salvas em: {io_cfg['output_csv']}")
    print(f"Tempo QA: {elapsed:.2f}s")

    return questions, answers, contexts


def unload_model(model_name: str):
    """Stop an Ollama model to free GPU/CPU memory.

    Args:
        model_name: Name of the Ollama model to stop.
    """
    try:
        subprocess.run(["ollama", "stop", model_name], check=True)
        print(f"Modelo {model_name} descarregado.")
    except subprocess.CalledProcessError:
        print(f"Aviso: falha ao descarregar modelo {model_name}")
    except FileNotFoundError:
        print("Aviso: comando 'ollama' nao encontrado no PATH")
