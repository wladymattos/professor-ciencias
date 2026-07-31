import os
import streamlit as st
import numpy as np
import base64
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

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
        "titulo": "🌌 Por que o céu é azul?",
        "sugestao_pergunta": "Por que a cor do céu é azul?",
        "texto": "Na aula sobre a Cor do Céu, explicamos que ele é azul por causa da dispersão da luz solar na atmosfera terrestre. A luz azul possui ondas mais curtas e se espalha muito mais do que as outras cores quando se choca com os gases do ar. Isso é conhecido fisicamente como Dispersão de Rayleigh.",
        "link": "https://youtube.com"
    },
    {
        "titulo": "🌱 Como funciona a Fotossíntese?",
        "sugestao_pergunta": "Como as plantas fazem fotossíntese?",
        "texto": "Na aula sobre a Fotossíntese, explicamos que as plantas usam a luz do Sol, a água que absorvem pelas raízes e o gás carbônico do ar para produzir glicose (seu alimento) e liberar oxigênio puro de volta para a atmosfera. A clorofila é a responsável por captar essa luz solar e dá a cor verde às folhas.",
        "link": "https://youtube.com"
    },
    {
        "titulo": "🪐 O Sistema Solar",
        "sugestao_pergunta": "Quais são os planetas do sistema solar?",
        "texto": "Na aula sobre o Sistema Solar, explicamos que o Sol fica no centro e oito planetas giram ao seu redor. Os quatro mais próximos são rochosos (Mercúrio, Vênus, Terra e Marte) e os quatro mais distantes são gigantes gasosos (Júpiter, Saturno, Urano e Netuno).",
        "link": "https://youtube.com"
    }
]

# Inicializa as matrixes matemáticas de busca semântica local
@st.cache_resource
def inicializar_busca_local():
    textos = [aula["texto"] for aula in AULAS_DO_CANAL]
    metadados = [{"source": aula["link"]} for aula in AULAS_DO_CANAL]
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    matriz_vetores = model_embedding.encode(textos, convert_to_numpy=True)
    return model_embedding, textos, metadados, matriz_vetores

model_embedding, lista_textos, lista_metas, matriz_vetores = inicializar_busca_local()

# ==============================================================================
# 🔑 GERENCIAMENTO DA CHAVE API DO ALUNO
# ==============================================================================
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False
    st.session_state.user_key = ""

# Se o usuário NÃO inseriu a chave, mostra apenas a tela de entrada da chave
if not st.session_state.autenticado:
    st.title("🧬 Portal do Aluno")
    st.markdown("Insira sua **Chave API do Gemini** para acessar o robô professor.")
    
    chave_api = st.text_input("🔑 Chave API do Gemini", type="password", placeholder="Cole aqui o seu código do Google AI Studio...")
    
    if st.button("🚪 Acessar o Professor"):
        if chave_api.strip() == "":
            st.warning("⚠️ Cole uma chave válida para continuar!")
        elif len(chave_api.strip()) < 10:
            st.error("❌ Esta chave de API está muito curta.")
        else:
            st.session_state.autenticado = True
            st.session_state.user_key = chave_api.strip()
            st.rerun()
    st.stop()

# ==============================================================================
# 🤖 INTERFACE PRINCIPAL DO CHAT (Acessível após inserir a chave)
# ==============================================================================
st.title("🧬 Robô Professor de Ciências")
st.subheader("Tire suas dúvidas com base em nossas videoaulas!")
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
# 🧼 BARRA LATERAL (Desconectar, Atalhos, PDFs e Limpeza)
# ==============================================================================
with st.sidebar:
    st.markdown("<h3 style='text-align: center; color: #2a5c4d;'>👤 Aluno Conectado</h3>", unsafe_allow_html=True)
    
    if st.button("🚪 Trocar Chave API"):
        st.session_state.autenticado = False
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
    
    caminho_pdf1 = "materiais/apostila_ciencias.pdf"
    if os.path.exists(caminho_pdf1):
        with open(caminho_pdf1, "rb") as file:
            st.download_button(
                label="📥 Baixar Apostila Geral (PDF)",
                data=file,
                file_name="Apostila_de_Ciencias.pdf",
                mime="application/pdf"
            )
            
    caminho_pdf2 = "materiais/exercicios_fotossintese.pdf"
    if os.path.exists(caminho_pdf2):
        with open(caminho_pdf2, "rb") as file:
            st.download_button(
                label="📝 Baixar Exercícios - Fotossíntese",
                data=file,
                file_name="Exercicios_Fotossintese.pdf",
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

caixa_entrada = st.chat_input("Digite sua dúvida aqui...")
pergunta = caixa_entrada if caixa_entrada else pergunta_clicada

if pergunta:
    with st.chat_message("user", avatar="🧑‍🎓"):
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

    # Limpeza preventiva de chaves globais do sistema antes de instanciar a IA
    if "GOOGLE_API_KEY" in os.environ: del os.environ["GOOGLE_API_KEY"]
    if "GEMINI_API_KEY" in os.environ: del os.environ["GEMINI_API_KEY"]

    with st.chat_message("assistant", avatar="🤖"):
        try:
            # CORREÇÃO DEFINITIVA DE INDENTAÇÃO: Alinhamento perfeito de 12 espaços para todas as linhas dentro do try


