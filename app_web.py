import os
import streamlit as st
import numpy as np
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

# Inicializa e indexa o banco de dados direto na nuvem usando Numpy puro
@st.cache_resource
def inicializar_sistema_completo():
    documentos = []
    for url in URLS_YOUTUBE:
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["pt", "pt-BR"])
            documentos.extend(loader.load())
        except Exception as e:
            print(f"Erro ao ler vídeo: {e}")
            
    # Trava de segurança contra falhas de conexão com o YouTube
    if not documentos:
        textos = ["A cor do céu é azul por causa da dispersão da luz solar na atmosfera terrestre. A luz azul se espalha mais que as outras cores (Dispersão de Rayleigh)."]
        metadados = [{"source": "https://youtube.com"}]
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        blocos_texto = text_splitter.split_documents(documentos)
        textos = [doc.page_content for doc in blocos_texto]
        metadados = [doc.metadata for doc in blocos_texto]
    
    # Cria os embeddings em matrizes matemáticas rápidas
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    matriz_vetores = model_embedding.encode(textos, convert_to_numpy=True)
    
    # CORREÇÃO: Tenta capturar a chave tanto do ambiente global quanto dos segredos do Streamlit
    chave_api = os.getenv("GOOGLE_API_KEY")
    if not chave_api and "GOOGLE_API_KEY" in st.secrets:
        chave_api = st.secrets["GOOGLE_API_KEY"]
        
    if not chave_api:
        st.error("⚠️ Chave GOOGLE_API_KEY não foi encontrada nas configurações do Streamlit Cloud!")
        st.stop()
    
    # Conecta ao cliente oficial do Google Gemini
    ai_client = genai.Client(
        api_key=chave_api,
        http_options=types.HttpOptions(api_version="v1")
    )
    
    return model_embedding, textos, metadados, matriz_vetores, ai_client

# Carrega as estruturas matemáticas
model_embedding, lista_textos, lista_metas, matriz_vetores, ai_client = inicializar_sistema_completo()

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

    # 1. Busca Semântica Matemática
    vetor_pergunta = model_embedding.encode([pergunta], convert_to_numpy=True)
    
    # Calcula a similaridade de cosseno
    scores = np.dot(matriz_vetores, vetor_pergunta.T).flatten()
    melhor_indice = int(np.argmax(scores))
    
    # Extrai o documento e link correspondente
    texto_encontrado = lista_textos[melhor_indice]
    meta_encontrada = lista_metas[melhor_indice]
    url_para_abrir = meta_encontrada.get("source", None) if meta_encontrada else None
    
    contexto_formatado = f"Conteúdo da aula: {texto_encontrado}\nLink do Vídeo: {url_para_abrir}"

    with st.chat_message("assistant"):
        try:
            config_ia = types.GenerateContentConfig(
                system_instruction=system_prompt + f"\n\nContexto:\n{contexto_formatado}",
                temperature=0.2
            )
            
            response = ai_client.models.generate_content(
                model="gemini-1.5-flash",
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
