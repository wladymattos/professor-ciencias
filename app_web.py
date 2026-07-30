import os
import streamlit as st
import numpy as np
import base64
import requests
from sentence_transformers import SentenceTransformer

# Configuração da página Web
st.set_page_config(
    page_title="Robô Professor de Ciências", 
    page_icon="🧬",
    layout="centered"
)

# ==============================================================================
# 🎨 DESIGN PREMIUM (IMAGEM OU DEGRADÊ DE FUNDO)
# ==============================================================================
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

img_base64 = get_base64_image("fundo.jpg")
if img_base64:
    css_fundo = f"""
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    """
else:
    css_fundo = ".stApp { background: linear-gradient(135deg, #eef5f3 0%, #dbe7e4 100%) !important; background-attachment: fixed; }"

st.markdown(f"""
    <style>
        {css_fundo}
        h1, h2, h3 {{ color: #1e3d33 !important; font-family: 'Helvetica Neue', Arial, sans-serif; font-weight: 700; }}
        .stButton>button {{ border-radius: 12px !important; background-color: #2a5c4d !important; color: white !important; border: none !important; width: 100%; padding: 10px !important; font-weight: bold !important; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05); transition: all 0.3s ease; }}
        .stButton>button:hover {{ background-color: #1e3d33 !important; transform: translateY(-2px); box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.1); }}
        .stChatMessage {{ background-color: rgba(255, 255, 255, 0.85) !important; border-radius: 15px !important; padding: 15px !important; margin-bottom: 10px !important; box-shadow: 0px 4px 10px rgba(0,0,0,0.05) !important; backdrop-filter: blur(8px); }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 CADASTRE SUAS AULAS AQUI
# ==============================================================================
AULAS_DO_CANAL = [
{
        "titulo": "🌌 O que é química?",
        "sugestao_pergunta": "Explique o que é química?",
        "texto": " Na aula sobre o que é química, apresentamos a química como responsável pela composição de tudo que se conhece no mundo..",
        "link": " https://www.youtube.com/watch?v=SCPEWIVOFiM&t=22s "
    },
    {
        "titulo": "🌱 O que são partículas?",
        "sugestao_pergunta": "Quais são as partículas que formam a matéria?",
        "texto": "Na aula sobre partículas explicamos como são constituídos os átomos e as partículas que formam a matéria.",
        "link": " https://www.youtube.com/watch?v=lw9nPJH2X8c "
    },
    {
        "titulo": "🪐 Soluções químicas",
        "sugestao_pergunta": "Como se formam as soluções químicas?",
        "texto": " Na aula sobre soluções explicamos o que é uma solução e como ela é formada.",
        "link": " https://www.youtube.com/watch?v=QT1osnLDjjA&t=8s "
    }
]
# Inicializa as matrizes matemáticas de busca semântica local
@st.cache_resource
def inicializar_busca_local():
    textos = [aula["texto"] for aula in AULAS_DO_CANAL]
    metadados = [{"source": aula["link"]} for aula in AULAS_DO_CANAL]
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    matriz_vetores = model_embedding.encode(textos, convert_to_numpy=True)
    return model_embedding, textos, metadados, matriz_vetores

model_embedding, lista_textos, lista_metas, matriz_vetores = inicializar_busca_local()

# ==============================================================================
# 🔑 GERENCIAMENTO DE SESSÃO E LOGIN
# ==============================================================================
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False
    st.session_state.user_email = ""
    st.session_state.user_key = ""

# Se o usuário NÃO estiver logado, mostra a tela de login
if not st.session_state.autenticado:
    st.title("🧬 Área de Login do Aluno")
    st.markdown("Por favor, entre com suas credenciais para acessar o robô professor.")
    
    email = st.text_input("📧 E-mail do Aluno", placeholder="exemplo@email.com")
    chave_api = st.text_input("🔑 Senha (Sua Chave API do Gemini)", type="password", placeholder="AIzaSy... ou GEMINI_...")
    
    if st.button("🚪 Entrar no Chat"):
        if email.strip() == "" or chave_api.strip() == "":
            st.warning("⚠️ Preencha todos os campos para continuar!")
        elif len(chave_api.strip()) < 10:
            st.error("❌ Esta chave de API está muito curta para ser válida. Verifique se copiou o código completo.")
        else:
            # CORREÇÃO DA URL: Incluído o 'models/' obrigatório na URL do teste de login
            url_teste = "https://googleapis.com"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": str(chave_api.strip())
            }
            payload = {"contents": [{"parts": [{"text": "oi"}]}]}
            
            try:
                response = requests.post(url_teste, json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    st.session_state.autenticado = True
                    st.session_state.user_email = email
                    st.session_state.user_key = chave_api.strip()
                    st.rerun()
                else:
                    try:
                        erro_msg = response.json().get("error", {}).get("message", "Chave inválida ou sem cota.")
                    except:
                        erro_msg = f"Erro no servidor do Google (Código {response.status_code})"
                    st.error(f"❌ Erro de Autenticação: {erro_msg}")
            except Exception as e:
                st.error(f"❌ Falha de conexão com os servidores do Google: {e}")
    st.stop()

# ==============================================================================
# 🤖 INTERFACE PRINCIPAL DO CHAT (Acessível após o login)
# ==============================================================================
st.title("🧬 Robô Professor de Ciências")
st.subheader(f"Olá, {st.session_state.user_email}! Tire suas dúvidas com base em nossas videoaulas.")
st.markdown("---")

system_prompt = (
    "Você é um robô professor de ciências altamente didático, paciente e divertido.\n"
    "Use APENAS o contexto das nossas aulas fornecido abaixo para responder à pergunta do aluno.\n"
    "Se o contexto não contiver a resposta, responda EXATAMENTE com a frase: 'Ih, eu ainda não gravei uma aula sobre esse assunto no canal!'.\n"
    "Se responder à pergunta usando o contexto, avise ao aluno que um link clicável e o player da aula foram disponibilizados abaixo para ele assistir."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

pergunta_clicada = None

# ==============================================================================
# 🧼 BARRA LATERAL (Logout, Atalhos, PDFs e Limpeza)
# ==============================================================================
with st.sidebar:
    st.markdown(f"<h3 style='text-align: center; color: #2a5c4d;'>👤 Aluno: {st.session_state.user_email}</h3>", unsafe_allow_html=True)
    
    if st.button("🚪 Sair da Conta"):
        st.session_state.autenticado = False
        st.session_state.user_email = ""
        st.session_state.user_key = ""
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 🎥 Aulas Disponíveis")
    st.write("Clique em uma aula para perguntar ao robô:")
    for aula in AULAS_DO_CANAL:
        if st.button(aula["titulo"], key=aula["titulo"]):
            pergunta_clicada = aula["sugestao_pergunta"]

    st.markdown("---")
    st.markdown("### 📚 Materiais de Apoio")
    
     # Exemplo de Botão de Download 1: EBS
    caminho_pdf1 = "materiais/Ensino_Baseado_Simulacao.pdf"
    if os.path.exists(caminho_pdf1):
        with open(caminho_pdf1, "rb") as file:
            st.download_button(
                label="📥 Baixar Ensino Baseado em Simulação (PDF)",
                data=file,
                file_name="Ensino_Baseado_Simulacao.pdf",
                mime="application/pdf"
            )
            
    # Exemplo de Botão de Download 2: Partículas
    caminho_pdf2 = "materiais/Particulas.pdf"
    if os.path.exists(caminho_pdf2):
        with open(caminho_pdf2, "rb") as file:
            st.download_button(
                label="📝 Baixar Partículas",
                data=file,
                file_name="Particulas.pdf",
                mime="application/pdf"
            )

    # Exemplo de Botão de Download 3: Teoria ácido-base de Lewis
    caminho_pdf3 = "materiais/Teoria_acido_base_Lewis.pdf"
    if os.path.exists(caminho_pdf3):
        with open(caminho_pdf3, "rb") as file:
            st.download_button(
                label="📥 Teoria ácido-base de Lewis",
                data=file,
                file_name="Teoria_acido_base_Lewis.pdf",
                mime="application/pdf"
            )

    st.markdown("---")
    if st.button("🗑️ Limpar Conversa (Recomeçar)"):
        st.session_state.messages = []
        st.rerun()
# ==============================================================================

if len(st.session_state.messages) == 0:
    st.info("👋 Escolha uma das aulas na barra lateral ou digite sua dúvida aqui embaixo. Eu vou te explicar o assunto e carregar o vídeo correspondente!")

for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])



