import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Configuração da página Web
st.set_page_config(page_title="🤖 Professor de Ciências AI", page_icon="🧬")
st.title("🤖 Robô Professor de Ciências")
st.subheader("Tire suas dúvidas com base em nossas videoaulas!")

# Inicializa o cérebro do robô (Cache para carregar apenas uma vez)
@st.cache_resource
def carregar_modelos():
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="aulas_ciencias")
    # Puxa a chave das variáveis de ambiente da hospedagem
    ai_client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
        http_options=types.HttpOptions(api_version="v1")
    )
    return model_embedding, collection, ai_client

try:
    model_embedding, collection, ai_client = carregar_modelos()
except Exception as e:
    st.error(f"Erro ao carregar o banco de dados: {e}. Certifique-se de que a pasta 'chroma_db' está no projeto.")

system_prompt = (
    "Você é um robô professor de ciências altamente didático, paciente e divertido.\n"
    "Use APENAS o contexto das nossas aulas fornecido abaixo para responder à pergunta do aluno.\n"
    "Se o contexto não contiver a resposta, diga de forma amigável: 'Ih, eu ainda não gravei uma aula sobre esse assunto no canal!'."
)

# Inicializa o histórico de mensagens na tela
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de entrada para o aluno digitar
if pergunta := st.chat_input("Pergunte algo sobre a nossa aula (ex: Por que o céu é azul?)"):
    # Mostra a pergunta do aluno no chat
    with st.chat_message("user"):
        st.markdown(pergunta)
    st.session_state.messages.append({"role": "user", "content": pergunta})

    # 1. Busca local no banco ChromaDB
    vetor_pergunta = model_embedding.encode(pergunta).tolist()
    resultados = collection.query(query_embeddings=[vetor_pergunta], n_results=1)
    
    contexto_formatado = ""
    url_para_abrir = None
    
    if resultados and resultados.get('documents') and resultados.get('metadatas'):
        lista_docs = resultados['documents'][0] if resultados['documents'] else []
        lista_metas = resultados['metadatas'][0] if resultados['metadatas'] else []
        
        if lista_docs and lista_metas:
            doc = lista_docs[0]
            meta = lista_metas[0]
            url_para_abrir = meta.get("source", None)
            contexto_formatado = f"Conteúdo da aula: {doc}\nLink do Vídeo: {url_para_abrir}"

    # 2. Gera a resposta com o Gemini
    with st.chat_message("assistant"):
        try:
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=pergunta,
                config={"system_instruction": system_prompt + f"\n\nContexto:\n{contexto_formatado}"}
            )
            resposta_final = response.text
            st.markdown(resposta_final)
            
            # 3. ATIVAÇÃO DO VÍDEO NA WEB: Se achar o link, exibe um botão clicável e incorpora o vídeo
            if url_para_abrir and "youtube.com" in url_para_abrir:
                st.video(url_para_abrir) # Incorpora o player do vídeo direto na página!
                st.success(f"🎬 Aula recomendada aberta acima!")
                
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})
            
        except Exception as e:
            st.error(f"Erro ao acionar a IA: {e}")
