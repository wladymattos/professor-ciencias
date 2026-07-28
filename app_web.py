import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuração da página Web
st.set_page_config(page_title="🤖 Professor de Ciências AI", page_icon="🧬")
st.title("🤖 Robô Professor de Ciências")
st.subheader("Tire suas dúvidas com base em nossas videoaulas!")

# LINKS DO SEU CANAL
URLS_YOUTUBE = [
    "https://youtube.com"
]

# Inicializa e indexa o banco de dados direto na nuvem de forma automática
@st.cache_resource
def inicializar_sistema_completo():
    documentos = []
    for url in URLS_YOUTUBE:
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["pt", "pt-BR"])
            documentos.extend(loader.load())
        except Exception as e:
            print(f"Erro ao ler vídeo: {e}")
            
    # TRAVA DE SEGURANÇA: Se a lista estiver vazia por erro do YouTube, cria um texto padrão
    if not documentos:
        textos = ["A cor do céu é azul por causa da dispersão da luz solar na atmosfera terrestre. A luz azul se espalha mais que as outras cores."]
        metadados = [{"source": "https://youtube.com"}]
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        blocos_texto = text_splitter.split_documents(documentos)
        textos = [doc.page_content for doc in blocos_texto]
        metadados = [doc.metadata for doc in blocos_texto]
    
    # Cria os embeddings locais na nuvem
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    vetores = model_embedding.encode(textos).tolist()
    
    # Inicializa o banco de dados ChromaDB temporário no servidor
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection(name="aulas_ciencias")
    
    ids = [f"id_{i}" for i in range(len(textos))]
    collection.add(embeddings=vetores, documents=textos, metadatas=metadados, ids=ids)
    
    # Conecta ao cliente oficial do Google Gemini forçando a versão de produção v1
    ai_client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
        http_options=types.HttpOptions(api_version="v1")
    )
    
    return model_embedding, collection, ai_client

# Executa o carregador automático
model_embedding, collection, ai_client = inicializar_sistema_completo()

system_prompt = (
    "Você é um robô professor de ciências altamente didático, paciente e divertido.\n"
    "Use APENAS o contexto das nossas aulas fornecido abaixo para responder à pergunta do aluno.\n"
    "Se o contexto não contiver a resposta, diga de forma amigável: 'Ih, eu ainda não gravei uma aula sobre esse assunto no canal!'."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pergunta := st.chat_input("Pergunte algo sobre a nossa aula (ex: Por que o céu é azul?)"):
    with st.chat_message("user"):
        st.markdown(pergunta)
    st.session_state.messages.append({"role": "user", "content": pergunta})

    # Busca no banco de dados criado na memória
    vetor_pergunta = model_embedding.encode(pergunta).tolist()
    resultados = collection.query(query_embeddings=[vetor_pergunta], n_results=1)
    
    contexto_formatado = ""
    url_para_abrir = None
    
    if resultados and resultados.get('documents') and resultados.get('metadatas'):
        # Garante o desempacotamento correto tirando a primeira camada de listas do Chroma
        lista_docs = resultados['documents'][0] if resultados['documents'] else []
        lista_metas = resultados['metadatas'][0] if resultados['metadatas'] else []
        
        if lista_docs and lista_metas:
            doc = lista_docs[0] if isinstance(lista_docs, list) else lista_docs
            meta = lista_metas[0] if isinstance(lista_metas, list) else lista_metas
            url_para_abrir = meta.get("source", None) if isinstance(meta, dict) else None
            contexto_formatado = f"Conteúdo da aula: {doc}\nLink do Vídeo: {url_para_abrir}"

    with st.chat_message("assistant"):
        try:
            # CORREÇÃO PARA A BIBLIOTECA GOOGLE-GENAI DO PYTHON 3.14
            # Na nova estrutura, as instruções do sistema entram em um formato de configuração próprio
            config_ia = types.GenerateContentConfig(
                system_instruction=system_prompt + f"\n\nContexto:\n{contexto_formatado}",
                temperature=0.2
            )
            
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=pergunta,
                config=config_ia
            )
            
            resposta_final = response.text
            st.markdown(resposta_final)
            
            if url_para_abrir and "youtube.com" in url_para_abrir:
                st.video(url_para_abrir)
                st.success(f"🎬 Aula recomendada aberta acima!")
                
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})
            
        except Exception as e:
            st.error(f"Erro ao acionar a IA: {e}")
