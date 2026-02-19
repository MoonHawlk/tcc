# Instale as dependências necessárias:
# pip install langchain transformers torch unstructured langchain-huggingface chromadb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline  # Novo pacote
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ----------------------------
# 1. Configuração do modelo LLM com Qwen/Qwen2.5-Math-1.5B
# ----------------------------
if torch.cuda.is_available():
    print("GPU detectada")
else:
    print("GPU não encontrada")

model_name = "Qwen/Qwen2.5-Math-1.5B"

# Carrega o tokenizer e o modelo com torch_dtype=float16 para otimizar VRAM
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_config = GenerationConfig(
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    max_new_tokens=128
)
model.generation_config = gen_config

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)

teste = hf_pipeline("Olá, mundo!")
print("\nTeste simples de geração:")
print(teste[0]["generated_text"])

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ----------------------------
# 2. Carregando e preparando o conteúdo da página
# ----------------------------
url = "https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural"
loader = UnstructuredURLLoader(urls=[url])
docs = loader.load()

if not docs:
    print("Nenhum documento foi carregado!")
else:
    print("\nDocumento carregado com sucesso. Exibindo os primeiros 200 caracteres do primeiro documento:")
    print(docs[0].page_content[:200])

text_splitter = CharacterTextSplitter(separator=" ", chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(docs)

print(f"\nNúmero de chunks criados: {len(docs)}")
for i, doc in enumerate(docs[:3]):
    print(f"\nChunk {i} (primeiros 200 caracteres):")
    print(doc.page_content[:200])

# ----------------------------
# 3. Criando o vetor de similaridade com Chroma
# ----------------------------
# Utiliza embeddings (neste exemplo, sentence-transformers) para converter os chunks em vetores
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)  # Chroma é compatível com Python 3.12
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ----------------------------
# 4. Configurando o chain de QA RAG com RetrievalQA
# ----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  # Pode ser ajustado para "stuff" se preferir
    retriever=retriever,
    return_source_documents=True  # Opcional para verificar os documentos recuperados
)

# ----------------------------
# 5. Realizando uma pergunta sobre o conteúdo da página
# ----------------------------
pergunta = "Quando começou a historia do PLN?"
print("\nRealizando a pergunta:")
print(pergunta)

output = qa_chain.invoke({"query": pergunta})
resposta = output["result"]

print("\nPergunta:", pergunta)
print("Resposta:", resposta)

