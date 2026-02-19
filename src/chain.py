"""QA chain construction with optional few-shot prompting."""


def build_qa_chain(llm, retriever, cfg: dict):
    """Build a RetrievalQA chain.

    Args:
        llm: LangChain LLM instance.
        retriever: LangChain retriever instance.
        cfg: Full pipeline configuration dict.

    Returns:
        RetrievalQA chain ready for .invoke() calls.
    """
    from langchain.chains import RetrievalQA

    chain_cfg = cfg["chain"]
    use_few_shot = chain_cfg.get("few_shot", False)

    kwargs = {}
    if use_few_shot:
        kwargs["chain_type_kwargs"] = {"prompt": _build_few_shot_prompt()}
        print("Few-shot prompting ativado.")

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_cfg["type"],
        retriever=retriever,
        return_source_documents=True,
        **kwargs,
    )


def _build_few_shot_prompt():
    """Build the few-shot prompt template with predefined examples."""
    from langchain.prompts import FewShotPromptTemplate, PromptTemplate

    examples = [
        {
            "question": "Quando comecou a historia do PLN?",
            "context": "O PLN comecou na decada de 1950...",
            "answer": "Na decada de 1950.",
        },
        {
            "question": "O que e PLN?",
            "context": "PLN e a area que estuda a interacao entre computadores e a linguagem humana.",
            "answer": "PLN estuda a interacao entre computadores e linguagem humana.",
        },
    ]

    example_prompt = PromptTemplate(
        input_variables=["question", "context", "answer"],
        template="Pergunta: {question}\nContexto: {context}\nResposta: {answer}\n",
    )

    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Voce e um assistente que responde perguntas com base em contexto.\n",
        suffix="\nPergunta: {question}\nContexto: {context}\nResposta:",
        input_variables=["question", "context"],
    )
