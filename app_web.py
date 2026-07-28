import os
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Configuração da página Web
st.set_page_config(page_title="🤖 Professor de Ciências AI", page_icon="🧬")
st.title("🤖 Robô Professor de Ciências")
st.subheader("Tire suas dúvidas com base em nossas videoaulas!")

# ==============================================================================
# 🧠 CADASTRE SUAS AULAS AQUI (Texto resumido + Link limpo de compartilhamento)
# ==============================================================================
AULAS_DO_CANAL = [
    {
        "texto": "Na aula sobre o que é química, apresentamos a química como responsável pela composição de tudo que se conhece no mundo.",
        "link": "https://www.youtube.com/watch?v=SCPEWIVOFiM&t=22s"
    },
    {
        "texto": "Na aula sobre partículas explicamos como são consituídos os átomos e as partículas que formam a matéria.",
        "link": "https://www.youtube.com/watch?v=lw9nPJH2X8c"
    },
    {
        "texto": "Na aula sobre soluções explicamos o que são soluções químicas e como elas são formadas.",
        "link": "https://www.youtube.com/watch?v=QT1osnLDjjA&t=8s"
    }

]
# ==============================================================================

@st.cache_resource
def inicializar_sistema_completo():
    textos = [aula["texto"] for aula in AULAS_DO_CANAL]
    metadados = [{"source": aula["link"]} for aula in AULAS_DO_CANAL]
    
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    matriz_vetores = model_embedding.encode(textos, convert_to_numpy=True)
    
    chave_api = os.getenv("GOOGLE_API_KEY")
    if not chave_api and "GOOGLE_API_KEY" in st.secrets:
        chave_api = st.secrets["GOOGLE_API_KEY"]
        
    if not chave_api:
        st.error("⚠️ Chave GOOGLE_API_KEY não configurada nos Secrets!")
        st.stop()
        
    ai_client = genai.Client(api_key=chave_api)
    return model_embedding, textos, metadados, matriz_vetores, ai_client

model_embedding, lista_textos, lista_metas, matriz_vetores, ai_client = inicializar_sistema_completo()

system_prompt = (
    "Você é um robô professor de ciências altamente didático, paciente e divertido.\n"
    "Use APENAS o contexto das nossas aulas fornecido abaixo para responder à pergunta do aluno.\n"
    "Se o contexto não contiver a resposta, responda EXATAMENTE com a frase: 'Ih, eu ainda não gravei uma aula sobre esse assunto no canal!'.\n"
    "Se responder à pergunta usando o contexto, avise ao aluno que um link clicável e o player da aula foram disponibilizados abaixo para ele assistir."
)

# Inicializa o histórico de mensagens na tela
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# 🧼 ADIÇÃO: BARRA LATERAL COM BOTÃO PARA LIMPAR O CHAT
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Opções do Chat")
    if st.button("🧹 Limpar Tela (Nova Pergunta)"):
        st.session_state.messages = []  # Apaga todas as mensagens da memória
        st.rerun()  # Recarrega a página com a tela limpa imediatamente
# ==============================================================================

# Exibe as mensagens armazenadas na sessão
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pergunta := st.chat_input("Pergunte algo sobre a nossa aula (ex: Como funciona a fotossíntese?)"):
    with st.chat_message("user"):
        st.markdown(pergunta)
    st.session_state.messages.append({"role": "user", "content": pergunta})

    # Busca Semântica Local
    vetor_pergunta = model_embedding.encode([pergunta], convert_to_numpy=True)
    scores = np.dot(matriz_vetores, vetor_pergunta.T).flatten()
    melhor_indice = int(np.argmax(scores))
    
    if scores[melhor_indice] > 0.35:
        texto_encontrado = lista_textos[melhor_indice]
        meta_encontrada = lista_metas[melhor_indice]
        url_para_abrir = meta_encontrada.get("source", None)
    else:
        texto_encontrado = "Assunto não coberto pelas aulas cadastradas."
        url_para_abrir = None

    contexto_formatado = f"Conteúdo da aula: {texto_encontrado}\nLink do Vídeo: {url_para_abrir}"

    with st.chat_message("assistant"):
        try:
            config_ia = types.GenerateContentConfig(
                system_instruction=system_prompt + f"\n\nContexto:\n{contexto_formatado}",
                temperature=0.2
            )
            
            response = ai_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=pergunta,
                config=config_ia
            )
            
            resposta_final = response.text
            st.markdown(resposta_final)
            
            if url_para_abrir and "youtube.com" in url_para_abrir:
                st.write(f"🔗 [Clique aqui para abrir o vídeo diretamente no YouTube]({url_para_abrir})")
                st.video(url_para_abrir)
                st.success("🎬 Player da aula carregado acima!")
                
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})
            
        except Exception as e:
            st.error(f"Erro ao acionar a IA: {e}")
