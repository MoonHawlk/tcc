"""RAGAS evaluation pipeline."""

import os


def run_ragas(questions: list, answers: list, contexts: list,
              embeddings, cfg: dict):
    """Evaluate QA results using RAGAS faithfulness and answer relevancy.

    Args:
        questions: List of question strings.
        answers: List of answer strings.
        contexts: List of context lists (one per question).
        embeddings: LangChain embeddings instance (reused from retriever).
        cfg: Full pipeline configuration dict.
    """
    import pandas as pd
    from dotenv import load_dotenv
    from datasets import Dataset
    from langchain.chat_models import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY nao encontrada no ambiente. "
                         "Crie um arquivo .env (veja .env.example).")

    ragas_cfg = cfg["ragas"]

    judger = ChatOpenAI(
        model_name=ragas_cfg["judge_model"],
        temperature=ragas_cfg["judge_temperature"],
        openai_api_key=openai_key,
    )

    run_config = RunConfig(
        max_workers=ragas_cfg["max_workers"],
        timeout=ragas_cfg["timeout"],
    )

    dataset = Dataset.from_pandas(pd.DataFrame({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }))

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(judger),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=run_config,
    )

    scores_path = cfg["io"]["scores_csv"]
    result.to_pandas().to_csv(scores_path, index=False)
    print(f"Scores salvos em: {scores_path}")
