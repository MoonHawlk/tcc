# Instale as dependências necessárias:
# pip install langchain transformers torch unstructured langchain-huggingface chromadb pandas ragas datasets

import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import FewShotPromptTemplate, PromptTemplate


# ---------------------------------------------------
# Estrutura da Arquitetura Atual
# ---------------------------------------------------
# [Config] → [LLM Init] → [Loader & Split] → [Embeddings & VectorStore] → [QA Chain] → [CSV I/O]
# Com isso, nós primeiro embeddamos a página web, ao qual posteriomente é consultada pelo nosso 
# Vetor de Similaridade e enriquecido o contexto, por fim, chegando a nossa LLM.
# ---------------------------------------------------

# ---------------------------------------------------
# Flags de configuração
# ---------------------------------------------------
# Ollama Mode (Executar modelos diretamente pelo Ollama)
# Few-Shot Mode (Descontinuado devido a necesside de aprimoramento a cada nova pergunta)

usar_ollama = True      # True para usar modelo via Ollama; False para HuggingFace
usar_few_shot = False   # True para usar few-shot prompt; False para não usar

# ---------------------------------------------------
# 1. Configuração do modelo LLM
# ---------------------------------------------------
# Via Hugging-Faces
if not usar_ollama:
    from langchain_huggingface import HuggingFacePipeline

    print("GPU detectada" if torch.cuda.is_available() else "GPU não encontrada")

    model_name = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    model.generation_config = GenerationConfig(
        temperature=0.3,
        top_k=30,
        top_p=0.8,
        max_new_tokens=128
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128
    )
    # teste simples
    out = hf_pipeline("Olá, mundo!")
    print("\nTeste simples de geração:", out[0]["generated_text"])

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

else:
    from langchain.llms import Ollama
    # Todos os modelos que seram testados, estão aqui.
    model_llama3_1 = "llama3.1:latest" 
    model_dolphin3 = "dolphin3:latest"
    model_zephyr = "zephyr:latest"
    model_sailor2 = "sailor2:latest" # Removido posteriormente por falta de benchmarks
    model_mistral = "mistral:latest"

    # Estes dois modelos são modelos usados para possiveis métricas futuras com Reasoning
    model_deepseek_r1_llama = "deepseek-r1:8b"
    model_gemma3_12b = "gemma3:12b"

    modelo_ollama = model_llama3_1
    llm = Ollama(model=modelo_ollama, 
                       temperature=0.1,  # Respostas mais conservadoras
                       top_k=30,         # Limita a diversidade dos tokens
                       top_p=0.8,        # Controla a probabilidade acumulada dos tokens escolhidos
                       )
    print("Usando modelo via Ollama:", modelo_ollama)

# ---------------------------------------------------
# 2. Carregando e preparando o conteúdo da página
# ---------------------------------------------------


# Página em Português Brasileiro escolhida
url_ptbr_pln_br = "https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural"

# As quatro páginas em Inglês escolhidas
url_en_nlp = "https://en.wikipedia.org/wiki/Natural_language_processing"
url_en_brazil = "https://en.wikipedia.org/wiki/Brazil"
url_en_new_gpt = "https://en.wikipedia.org/wiki/GPT-4.5"
url_en_new_deepseek = "https://en.wikipedia.org/wiki/DeepSeek_%28chatbot%29"

# Carregamento do documento via LangChain
loader = UnstructuredURLLoader(urls=[url_en_new_gpt])
docs = loader.load()
if not docs:
    raise RuntimeError("Nenhum documento foi carregado!")

print("\nDocumento carregado. Exemplo de 200 caracteres:")
print(docs[0].page_content[:200])

# Preparação via LangChain para formato de Embeddings
text_splitter = CharacterTextSplitter(separator=" ", chunk_size=512, chunk_overlap=50)
docs = text_splitter.split_documents(docs)
print(f"\nNúmero de chunks: {len(docs)}")

# ---------------------------------------------------
# 3. Criando o vetor de similaridade com Chroma
# ---------------------------------------------------
# Escolha do modelo de embeddings
# Neste caso o uso foi o all-Mini, apenas por testes previamente realiados.
# Ambos os modelos são bons e promissores

all_mini = "sentence-transformers/all-MiniLM-L6-v2"
snowflake = "Snowflake/snowflake-arctic-embed-l-v2.0"
embeddings = HuggingFaceEmbeddings(model_name=all_mini)
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------
# 4. (Opcional) Configurando prompt few-shot
# ---------------------------------------------------
if usar_few_shot:
    exemplos = [
        {"question": "Quando começou a história do PLN?",
         "context": "O PLN começou na década de 1950...",
         "answer": "Na década de 1950."},
        {"question": "O que é PLN?",
         "context": "PLN é a área que estuda a interação entre computadores e a linguagem humana.",
         "answer": "PLN estuda a interação entre computadores e linguagem humana."}
    ]
    prompt_exemplo = PromptTemplate(
        input_variables=["question", "context", "answer"],
        template="Pergunta: {question}\nContexto: {context}\nResposta: {answer}\n"
    )
    few_shot_prompt = FewShotPromptTemplate(
        examples=exemplos,
        example_prompt=prompt_exemplo,
        prefix="Você é um assistente que responde perguntas com base em contexto.\n",
        suffix="\nPergunta: {question}\nContexto: {context}\nResposta:",
        input_variables=["question", "context"]
    )

# ---------------------------------------------------
# 5. Configurando o chain de QA
# ---------------------------------------------------
# Ajuste realizado para validar o few-shot
# Somado a definição da chain_type "Refine", para garantir que ele tenha uma escolha individual de arquivos
# Não é necessario, porém assegura que a LLM tente responder da melhor forma que ela conseguir
# https://js.langchain.com/v0.1/docs/modules/chains/document/refine/

if usar_few_shot:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": few_shot_prompt}
    )
else:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=True
    )

# ---------------------------------------------------
# 6. Processando perguntas de um CSV e salvando respostas
# ---------------------------------------------------
csv_input = "perguntas.csv"    # Deve conter coluna "pergunta"
csv_output = "respostas.csv"

df = pd.read_csv(csv_input)
if "pergunta" not in df.columns:
    raise ValueError("Coluna 'pergunta' não encontrada no CSV.")

perguntas = df["pergunta"].tolist()
respostas = []
contexts_list = []

start_time = time.time()
for pergunta in perguntas:
    print("\nPergunta:", pergunta)
    out = qa_chain.invoke({"query": pergunta})
    respostas.append(out["result"])
    contexts_list.append([d.page_content for d in out["source_documents"]])
    print("Resposta:", respostas[-1])
elapsed = time.time() - start_time

# Salva respostas preliminares, antes do teste automatizado
df["resposta"] = respostas
df.to_csv(csv_output, index=False)
print(f"\nRespostas salvas em: {csv_output}")
print(f"Tempo QA: {elapsed:.2f}s")

import subprocess

# Limpa da memoria o modelo de LLM usado, para garantir que haverá memória
# Para um possivel teste de LLM via LLM local
def unload_model(model_name: str):
    """
    Descarrega (stop) o modelo do Ollama para liberar memória GPU/CPU.
    """
    # opção A: invocando o CLI do Ollama
    subprocess.run(["ollama", "stop", model_name], check=True)

unload_model(modelo_ollama)

# ---------------------------------------------------
# 7. Avaliação com RAGAS
# ---------------------------------------------------
from ragas.metrics import faithfulness, answer_relevancy
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas import evaluate
from dotenv import load_dotenv
import os

# Carrega as variáveis definidas no .env para o ambiente
load_dotenv()  

# Recupera a chave
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("Não encontrou OPENAI_API_KEY no ambiente")

# Configs Ragas
run_config = RunConfig(max_workers=2,
                       timeout=120)

llm_judger = ChatOpenAI(
    model_name="gpt-4.1-nano-2025-04-14",
    temperature=0.1,
    openai_api_key=openai_key
)

wrapped_llm        = LangchainLLMWrapper(llm_judger)
wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Prepare seu dataset igual antes
from datasets import Dataset
df_eval = pd.DataFrame({
    "question": perguntas,
    "answer":   respostas,
    "contexts": contexts_list
})
hf_dataset = Dataset.from_pandas(df_eval)

# Realiza a Pipeline para avaliar categorias de Fidelidade da Resposta e Relevancia Direta.
result = evaluate(
    dataset=hf_dataset,                         # Dataset (Conjunto de Perguntas e Respostas que nosso modelo gerou, assim como o contexto retornado)
    metrics=[faithfulness, answer_relevancy],   # Métricas escolhidas
    llm=wrapped_llm,                            # LLM que será o Juíz
    embeddings=wrapped_embeddings,              # Embeddings usados para o banco vetorial
    run_config=run_config
)

# Salva as respostas em um arquivo .csv
df_scores = result.to_pandas()
df_scores.to_csv("scores_ragas.csv", index=False)
print("Scores salvos em scores_ragas.csv")