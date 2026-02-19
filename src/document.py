"""Document loading and text splitting."""


def load_and_split(cfg: dict) -> list:
    """Load a web page and split it into chunks for embedding.

    Args:
        cfg: Full pipeline configuration dict.

    Returns:
        List of LangChain Document chunks.
    """
    from langchain.document_loaders import UnstructuredURLLoader
    from langchain.text_splitter import CharacterTextSplitter

    doc_cfg = cfg["document"]
    url = doc_cfg["url"]

    loader = UnstructuredURLLoader(urls=[url])
    docs = loader.load()

    if not docs:
        raise RuntimeError(f"Nenhum documento carregado de: {url}")

    print(f"\nDocumento carregado de: {url}")
    print(f"Preview: {docs[0].page_content[:200]}")

    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=doc_cfg["chunk_size"],
        chunk_overlap=doc_cfg["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks criados: {len(chunks)}")

    return chunks
