# Comparison of LLM Models for Data Retrieval from Static Pages Using RAG Architecture

## Overview
This work aims to present a comparative analysis of data retrieval from static pages using Large Language Models (LLMs) and a Retrieval-Augmented Generation (RAG) architecture. The study seeks to understand the nuances and challenges involved in choosing the most effective model for this task, highlighting the importance of robust and standardized evaluation methods.

Considering that, this repository demonstrates a pipeline for retrieving information from static web pages using Large Language Models (LLMs) combined with a Retrieval-Augmented Generation (RAG) architecture. It supports experimentation with both Ollama-served models and HuggingFace-hosted models, and includes automated evaluation of generated answers using the RAGAS framework.

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration Flags](#configuration-flags)  
- [Architecture](#architecture)  
- [How It Works](#how-it-works)  
  1. [Model Initialization](#1-model-initialization)  
  2. [Document Loading & Splitting](#2-document-loading--splitting)  
  3. [Embeddings & Vector Store](#3-embeddings--vector-store)  
  4. [(Optional) Few-Shot Prompting](#4-optional-few-shot-prompting)  
  5. [Building the QA Chain](#5-building-the-qa-chain)  
  6. [Batch Query Processing & CSV I/O](#6-batch-query-processing--csv-io)  
  7. [Automated Evaluation with RAGAS](#7-automated-evaluation-with-ragas)  
- [Input & Output Specifications](#input--output-specifications)  
- [Cleaning Up & Unloading Models](#cleaning-up--unloading-models)  
- [Notes & Tips](#notes--tips)  

## Prerequisites

- **Python** = 3.12.3 
- **CUDA-enabled GPU** (optional, but recommended for large HuggingFace & Ollama models)  
- An **Ollama** installation (if using Ollama-served models)  
- A valid **OpenAI API key** in your environment, named `OPENAI_API_KEY`  

## Installation

```bash
# Clone this repository
git clone https://github.com/MoonHawlk/tcc.git
cd tcc

# Install required Python packages
pip install uv
uv pip install langchain transformers torch unstructured langchain-huggingface chromadb pandas ragas datasets
```

## Configuration Flags

- `usar_ollama`  
  - **Type:** boolean  
  - **Default:** `True`  
  - **Description:**  
    - `True`: use Ollama-served models.  
    - `False`: use HuggingFace models via a local pipeline.  

- `usar_few_shot`  
  - **Type:** boolean  
  - **Default:** `False`  
  - **Description:**  
    - `True`: prepend a few-shot prompt template to each query (requires manual maintenance of examples).  
    - `False`: use a zero-shot refine chain.

## Architecture

```
[Config] → [LLM Init] → [Loader & Split] → [Embeddings & VectorStore] → [QA Chain] → [CSV I/O]
```
This linear pipeline ingests configuration flags, initializes the LLM, retrieves and preprocesses documents, builds a similarity search index, answers queries in bulk, and writes outputs to CSV.

## How It Works

### 1. Model Initialization

- **HuggingFace Mode** (`usar_ollama = False`):  
  1. Detect GPU availability.  
  2. Load `Qwen/Qwen2.5-Math-1.5B` with half-precision (`float16`) onto GPU/CPU.  
  3. Configure generation hyperparameters (temperature, top_k, top_p, etc.).  
  4. Wrap in a LangChain `HuggingFacePipeline` for downstream use.  
  5. Run a quick “Olá, mundo!” sanity check.  

- **Ollama Mode** (`usar_ollama = True`):  
  1. Define a shortlist of Ollama models (e.g., `llama3.1:latest`, `zephyr:latest`).  
  2. Select `modelo_ollama` (default: `zephyr:latest`).  
  3. Instantiate `langchain.llms.Ollama` with low temperature for conservative answers.  

Prints an initialization message indicating which path was taken.

### 2. Document Loading & Splitting

1. **URL Selection**  
   - Default target: GPT-4.5 Wikipedia page (`https://en.wikipedia.org/wiki/GPT-4.5`).  
2. **Loading**  
   - Use `UnstructuredURLLoader` to fetch raw page text.  
   - Error out if loading fails.  
3. **Splitting**  
   - Instantiate `CharacterTextSplitter` with:
     - `separator=" "`  
     - `chunk_size=512`  
     - `chunk_overlap=50`  
   - Split raw document into semantically coherent paragraphs/chunks.  

Logs the number of chunks generated.

### 3. Embeddings & Vector Store

1. **Embeddings Model**  
   - By default: `sentence-transformers/all-MiniLM-L6-v2`.  
2. **Indexing**  
   - Create a Chroma vector store from the document chunks.  
   - Wrap it as a retriever with `search_kwargs={"k": 3}` for top-3 similarity results.

### 4. (Optional) Few-Shot Prompting

When `usar_few_shot = True`:

- Define a small set of `[question, context, answer]` examples.  
- Build a `FewShotPromptTemplate` that prefixes each query.  
- Pass this template into the QA chain’s `chain_type_kwargs`.  

This helps steer the model but requires manual upkeep.

### 5. Building the QA Chain

- Use `RetrievalQA.from_chain_type` with:
  - `llm`: the initialized LLM  
  - `chain_type="refine"` for an iterative answer-refinement approach  
  - `retriever`: the Chroma retriever  
  - `return_source_documents=True` to capture provenance  
  - Optional `prompt` override for few-shot  

### 6. Batch Query Processing & CSV I/O

1. **Inputs**  
   - `perguntas.csv` must contain a column header `"pergunta"`.  
2. **Loop**  
   - Read all questions into a list.  
   - For each question:
     - Invoke `qa_chain` with `{"query": pergunta}`.  
     - Collect `result` (answer) and `source_documents` (contexts).  
3. **Outputs**  
   - Append answers to the DataFrame as a `"resposta"` column.  
   - Write to `respostas.csv`.  
   - Log total QA time.

### 7. Automated Evaluation with RAGAS

After QA is complete:

1. **Environment Setup**  
   - Load `.env` and retrieve `OPENAI_API_KEY`.  
2. **Judger LLM**  
   - Instantiate `ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14")` for evaluation.  
3. **Wrappers**  
   - Wrap both the judger LLM and embeddings in RAGAS-compatible adapters.  
4. **Dataset Preparation**  
   - Build a HuggingFace `Dataset` from questions, answers, and captured contexts.  
5. **Metrics**  
   - Evaluate using `faithfulness` and `answer_relevancy`.  
   - Save results to `scores_ragas.csv`.

## Input & Output Specifications

| File             | Role                                   |
| ---------------- | -------------------------------------- |
| `perguntas.csv`  | Input table with a `pergunta` column.  |
| `respostas.csv`  | QA answers appended to original table. |
| `scores_ragas.csv` | RAGAS evaluation scores.            |

## Cleaning Up & Unloading Models

- After processing, the script defines and calls:

  ```python
  def unload_model(model_name: str):
      subprocess.run(["ollama", "stop", model_name], check=True)
  ```

- Ensures GPU/CPU memory is freed before evaluation.

## Notes & Tips

- **Switching Models:** Toggle `usar_ollama` to compare local vs. Ollama-served models.  
- **Chunk Parameters:** Adjust `chunk_size` and `chunk_overlap` based on document length.  
- **Retriever Tuning:** Change `k` in `search_kwargs` to retrieve more or fewer context chunks.  
- **Prompt Engineering:** Use `usar_few_shot` judiciously; it can improve accuracy at the cost of maintenance.  
- **Evaluation Budget:** RAGAS evaluation may incur OpenAI API usage; monitor `timeout` and `max_workers` settings.
