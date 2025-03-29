import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

st.title("QA Interativo com Streamlit")

# Exibe se a GPU foi detectada
if torch.cuda.is_available():
    st.write("GPU detectada")
else:
    st.write("GPU não encontrada")


# ----------------------------
# 1. Carregamento do Modelo LLM
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_llm():
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

    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    return llm


llm = load_llm()


# ----------------------------
# 2. Carregamento e Preparação dos Documentos a partir do URL fornecido pelo usuário
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_documents(url):
    loader = UnstructuredURLLoader(urls=[url])
    docs = loader.load()
    if not docs:
        return None
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(docs)
    return docs


# Alteramos o parâmetro para "_docs" para evitar problemas de hash
@st.cache_resource(show_spinner=False)
def setup_retriever(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


# Entrada do URL pelo usuário
user_url = st.text_input(
    "Digite a URL do documento que deseja analisar:",
    "https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural"
)

if st.button("Carregar Documento"):
    with st.spinner("Carregando documento..."):
        docs = load_documents(user_url)
    if docs is None:
        st.error("Não foi possível carregar o documento a partir da URL fornecida.")
    else:
        st.success("Documento carregado com sucesso!")
        st.write("Exibindo os primeiros 200 caracteres do primeiro chunk:")
        st.write(docs[0].page_content[:200])

        # Configura o retriever e o chain de QA
        retriever = setup_retriever(docs)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",  # Pode ser ajustado para "stuff" se preferir
            retriever=retriever,
            return_source_documents=True  # Opcional: exibe os trechos dos documentos usados na resposta
        )

        # ----------------------------
        # 3. Realizando Pergunta sobre o Conteúdo Carregado
        # ----------------------------
        question = st.text_input("Digite sua pergunta sobre o conteúdo:")
        if st.button("Enviar Pergunta"):
            with st.spinner("Processando sua pergunta..."):
                output = qa_chain.invoke({"query": question})
                resposta = output["result"]
            st.markdown("### Resposta:")
            st.write(resposta)

            # Exibe os documentos fonte (opcional)
            source_docs = output.get("source_documents", [])
            if source_docs:
                st.markdown("### Documentos de Origem (trechos):")
                for i, doc in enumerate(source_docs):
                    st.write(f"**Documento {i + 1}:**")
                    st.write(doc.page_content[:300] + "...")
