# Instale as dependências necessárias:
# pip install langchain transformers torch unstructured langchain-huggingface chromadb pandas

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# ---------------------------------------------------
# Flags de configuração
# ---------------------------------------------------
usar_ollama = False      # True para usar modelo via Ollama; False para HuggingFace
usar_few_shot = False   # True para usar few-shot prompt; False para não usar (recomendado para perguntas diversas)

# ---------------------------------------------------
# 1. Configuração do modelo LLM
# ---------------------------------------------------
if not usar_ollama:
    from langchain_huggingface import HuggingFacePipeline

    if torch.cuda.is_available():
        print("GPU detectada")
    else:
        print("GPU não encontrada")

    model_name = "Qwen/Qwen2.5-Math-1.5B"

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
else:
    from langchain.llms import Ollama
    # Certifique-se de que o modelo escolhido está disponível (ex: via 'ollama pull llama3')
    modelo_ollama = "llama3"  # ou "mistral", "llama2", etc.
    llm = Ollama(model=modelo_ollama)
    print("Usando modelo via Ollama:", modelo_ollama)

# ---------------------------------------------------
# 2. Carregando e preparando o conteúdo da página
# ---------------------------------------------------
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

# ---------------------------------------------------
# 3. Criando o vetor de similaridade com Chroma
# ---------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------
# 4. (Opcional) Configurando o few-shot prompt
# ---------------------------------------------------
if usar_few_shot:
    exemplos = [
        {
            "question": "Quando começou a história do PLN?",
            "context": "O processamento de linguagem natural (PLN) começou a se desenvolver na década de 1950, com pesquisas pioneiras na área de inteligência artificial.",
            "answer": "A história do PLN começou na década de 1950."
        },
        {
            "question": "O que é PLN?",
            "context": "PLN, ou Processamento de Linguagem Natural, é uma área da inteligência artificial que se dedica a estudar a interação entre computadores e a linguagem humana.",
            "answer": "PLN é a área da inteligência artificial que estuda a interação entre computadores e a linguagem humana."
        }
    ]

    template_exemplo = (
        "Pergunta: {question}\n"
        "Contexto: {context}\n"
        "Resposta: {answer}\n"
    )
    prompt_exemplo = PromptTemplate(
        input_variables=["question", "context", "answer"],
        template=template_exemplo
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=exemplos,
        example_prompt=prompt_exemplo,
        prefix="Você é um assistente que responde perguntas com base em um contexto fornecido. Responda de forma concisa e direta.\n",
        suffix="\nPergunta: {question}\nContexto: {context}\nResposta:",
        input_variables=["question", "context"]
    )

# ---------------------------------------------------
# 5. Configurando o chain de QA
# ---------------------------------------------------
if usar_few_shot:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' concatena o contexto e o prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": few_shot_prompt}
    )
else:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# ---------------------------------------------------
# 6. Processando múltiplas perguntas de um CSV e salvando as respostas
# ---------------------------------------------------
csv_input = "perguntas.csv"   # arquivo CSV de entrada (deve conter a coluna "pergunta")
csv_output = "respostas.csv"   # arquivo CSV de saída

df = pd.read_csv(csv_input)
if "pergunta" not in df.columns:
    raise ValueError("A coluna 'pergunta' não foi encontrada no arquivo CSV.")

lista_perguntas = df["pergunta"].tolist()
lista_respostas = []

for pergunta in lista_perguntas:
    print("\nPergunta:", pergunta)
    output = qa_chain.invoke({"query": pergunta})
    resposta = output["result"]
    lista_respostas.append(resposta)
    print("Resposta:", resposta)

df["resposta"] = lista_respostas
df.to_csv(csv_output, index=False)
print("\nRespostas salvas em:", csv_output)
