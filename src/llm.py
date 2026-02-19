"""LLM initialization for Ollama and HuggingFace backends."""


def build_llm(cfg: dict) -> tuple:
    """Build an LLM instance from config.

    Returns:
        (llm, model_name) tuple.
    """
    backend = cfg.get("backend", "ollama")

    if backend == "ollama":
        return _build_ollama(cfg)
    elif backend == "huggingface":
        return _build_huggingface(cfg)
    else:
        raise ValueError(f"Backend desconhecido: {backend}")


def _build_ollama(cfg: dict) -> tuple:
    """Initialize an Ollama-served model via LangChain."""
    from langchain.llms import Ollama

    ollama_cfg = cfg["ollama"]
    model_name = ollama_cfg["model"]

    llm = Ollama(
        model=model_name,
        temperature=ollama_cfg["temperature"],
        top_k=ollama_cfg["top_k"],
        top_p=ollama_cfg["top_p"],
    )
    print(f"Usando modelo via Ollama: {model_name}")
    return llm, model_name


def _build_huggingface(cfg: dict) -> tuple:
    """Initialize a HuggingFace model via local pipeline.

    Heavy imports (torch, transformers) are deferred to here
    so they are only loaded when this backend is selected.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        pipeline,
    )
    from langchain_huggingface import HuggingFacePipeline

    hf_cfg = cfg["huggingface"]
    model_name = hf_cfg["model_name"]

    print("GPU detectada" if torch.cuda.is_available() else "GPU nao encontrada")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    model.generation_config = GenerationConfig(
        temperature=hf_cfg["temperature"],
        top_k=hf_cfg["top_k"],
        top_p=hf_cfg["top_p"],
        max_new_tokens=hf_cfg["max_new_tokens"],
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=hf_cfg["max_new_tokens"],
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print(f"Usando modelo via HuggingFace: {model_name}")
    return llm, model_name
