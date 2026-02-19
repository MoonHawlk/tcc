"""Embeddings and vector store retriever setup."""


def build_retriever(docs: list, cfg: dict) -> tuple:
    """Create embeddings, index documents in Chroma, and return a retriever.

    Args:
        docs: List of LangChain Document chunks.
        cfg: Full pipeline configuration dict.

    Returns:
        (retriever, embeddings) tuple. Embeddings are returned because
        the RAGAS evaluation step also needs them.
    """
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma

    emb_cfg = cfg["embeddings"]

    embeddings = HuggingFaceEmbeddings(model_name=emb_cfg["model_name"])
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": emb_cfg["retriever_k"]}
    )

    return retriever, embeddings
