import os
import streamlit as st
import numpy as np
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Configuração da página Web com título e ícone customizados
st.set_page_config(
    page_title="Robô Professor de Ciências", 
    page_icon="🧬",
    layout="centered"
)

# ==============================================================================
# 📄 FUNÇÃO PARA GERAR O PDF DA RESPOSTA DINAMICAMENTE
# ==============================================================================
def gerar_pdf_resposta(pergunta, resposta):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40, title="Resposta do Professor de Ciências")
    
    styles = getSampleStyleSheet()
    
    estilo_titulo = ParagraphStyle(
        'TituloPDF',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e3d33'),
        spaceAfter=15
    )
    estilo_pergunta = ParagraphStyle(
        'PerguntaPDF',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2a5c4d'),
        spaceAfter=15
    )
    estilo_corpo = ParagraphStyle(
        'CorpoPDF',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#333333'),
        spaceAfter=10
    )
    
    story = []
    story.append(Paragraph("🧬 Robô Professor de Ciências — Explicação", estilo_titulo))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Dúvida do Aluno:</b> {pergunta}", estilo_pergunta))
    story.append(Spacer(1, 10))
    
    linhas = resposta.split('\n')
    for linha in linhas:
        if linha.strip():
            story.append(Paragraph(linha.strip(), estilo_corpo))
            
    doc.build(story)
    buffer.seek(0)
    return buffer

# ==============================================================================
# 🖼️ FUNÇÃO COMPLEMENTAR: CONVERTE A IMAGEM DO GITHUB EM FUNDO SEGURO
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
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
    css_fundo = """
    .stApp {
        background: linear-gradient(135deg, #eef5f3 0%, #dbe7e4 100%) !important;
        background-attachment: fixed;
    }
    """

st.markdown(f"""
    <style>
        {css_fundo}
        h1, h2, h3 {{
            color: #1e3d33 !important;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-weight: 700;
        }}
        .stButton>button, .stDownloadButton>button {{
            border-radius: 12px !important;
            background-color: #2a5c4d !important;
            color: white !important;
            border: none !important;
            width: 100%;
            padding: 10px !important;
            font-weight: bold !important;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            background-color: #1e3d33 !important;
            transform: translateY(-2px);
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.1);
        }}
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 15px !important;
            padding: 15px !important;
            margin-bottom: 10px !important;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05) !important;
            backdrop-filter: blur(8px);
        }}
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Robô Professor de Ciências")
st.markdown("---")

# ==============================================================================
# 🧠 CADASTRE SUAS AULAS AQUI
# ==============================================================================
AULAS_DO_CANAL = [
    {
        "titulo": "🌌 O que é química?",
        "sugestao_pergunta": "Explique o que é química?",
        "texto": "Na aula sobre o que é química, apresentamos a química como responsável pela composição de tudo que se conhece no mundo..",
        "link": "https://youtube.com"
    },
    {
        "titulo": "🌱 O que são partículas?",
        "sugestao_pergunta": "Quais são as partículas que formam a matéria?",
        "texto": "Na aula sobre partículas explicamos como são constituídos os átomos e as partículas que formam a matéria.",
        "link": "https://youtube.com"
    },
    {
        "titulo": "🪐 Soluções químicas",
        "sugestao_pergunta": "Como se formam as soluções químicas?",
        "texto": "Na aula sobre soluções explicamos o que é uma solution e como ela é formada.",
        "link": "https://youtube.com"
    }
]

@st.cache_resource
def inicializar_sistema_completo():
    textos = [aula["texto"] for aula in AULAS_DO_CANAL]
    metadados = [{"source": aula["link"], "titulo": aula["titulo"]} for aula in AULAS_DO_CANAL]
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
    "Responda a qualquer dúvida ou conceito de ciências que o aluno trouxer da forma mais clara e educativa possível.\n"
    "Caso o contexto das nossas aulas fornecido abaixo seja diretamente relevante para o tema da pergunta, "
    "faça uma menção natural na sua resposta dizendo que há uma aula completa sobre esse tema gravada no canal."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pergunta_clicada" not in st.session_state:
    st.session_state.pergunta_clicada = None

# ==============================================================================
# 🧼 BARRA LATERAL
# ==============================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2a5c4d;'>📌 Painel do Aluno</h2>", unsafe_allow_html=True)
    st.markdown("### 🎥 Aulas Disponíveis")
    st.write("Clique em uma aula para perguntar ao robô:")
    for aula in AULAS_DO_CANAL:
        if st.button(aula["titulo"], key=f"btn_{aula['titulo']}"):
            st.session_state.pergunta_clicada = aula["sugestao_pergunta"]
            st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Materiais de Apoio")
    
    caminho_pdf1 = "materiais/Ensino_Baseado_Simulacao.pdf"
    if os.path.exists(caminho_pdf1):
        with open(caminho_pdf1, "rb") as file:
            st.download_button(label="📥 Baixar Ensino Baseado em Simulação (PDF)", data=file, file_name="Ensino_Baseado_Simulacao.pdf", mime="application/pdf", key="mat1")
            
    caminho_pdf2 = "materiais/Particulas.pdf"
    if os.path.exists(caminho_pdf2):
        with open(caminho_pdf2, "rb") as file:
            st.download_button(label="📝 Baixar Partículas", data=file, file_name="Particulas.pdf", mime="application/pdf", key="mat2")

    caminho_pdf3 = "materiais/Teoria_acido_base_Lewis.pdf"
    if os.path.exists(caminho_pdf3):
        with open(caminho_pdf3, "rb") as file:
            st.download_button(label="📥 Teoria ácido-base de Lewis", data=file, file_name="Teoria_acido_base_Lewis.pdf", mime="application/pdf", key="mat3")

    st.markdown("---")
    if st.button("🗑️ Limpar Conversa (Recomeçar)", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.pergunta_clicada = None
        st.rerun()

# Mensagem de boas-vindas estática
if not st.session_state.messages:
    st.info("👋 **Olá, cientista!** Escolha uma das aulas na barra lateral ou digite sua dúvida sobre qualquer assunto de Ciências abaixo!")

# Histórico de conversas
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "video_link" in message and message["video_link"]:
                st.markdown(f"🔗 **Aula recomendada:** [{message['video_titulo']}]({message['video_link']})")
                st.video(message["video_link"])
            pdf_data = gerar_pdf_resposta(st.session_state.messages[i-1]["content"], message["content"])
            st.download_button(label="📥 Baixar Resposta em PDF", data=pdf_data, file_name=f"resposta_ciencias_{i}.pdf", mime="application/pdf", key=f"dl_{i}")

# Input de chat
prompt_usuario = st.chat_input("Digite sua dúvida de ciências...")

if st.session_state.pergunta_clicada:
    prompt_usuario = st.session_state.pergunta_clicada
    st.session_state.pergunta_clicada = None

if prompt_usuario:
