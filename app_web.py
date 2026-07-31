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

# Configuração da página Web
st.set_page_config(page_title="Robô Professor de Ciências", page_icon="🧬", layout="centered")

# Função para gerar o PDF da resposta
def gerar_pdf_resposta(pergunta, resposta):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    estilo_titulo = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1e3d33'), spaceAfter=12)
    estilo_pergunta = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#2a5c4d'), spaceAfter=12)
    estilo_corpo = ParagraphStyle('C1', parent=styles['Normal'], fontSize=11, leading=16, textColor=colors.HexColor('#333333'), spaceAfter=8)
    
    story = [
        Paragraph("🧬 Robô Professor de Ciências — Resposta", estilo_titulo),
        Spacer(1, 10),
        Paragraph(f"<b>Dúvida do Aluno:</b> {pergunta}", estilo_pergunta),
        Spacer(1, 10)
    ]
    
    for linha in resposta.split('\n'):
        if linha.strip():
            story.append(Paragraph(linha.strip(), estilo_corpo))
            
    doc.build(story)
    buffer.seek(0)
    return buffer

# Conversor de imagem de fundo
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

img_base64 = get_base64_image("fundo.jpg")
css_fundo = f'.stApp {{ background-image: url("data:image/jpg;base64,{img_base64}"); background-size: cover; background-attachment: fixed; }}' if img_base64 else '.stApp { background: linear-gradient(135deg, #eef5f3 0%, #dbe7e4 100%) !important; }'

st.markdown(f"<style>{css_fundo} h1, h2, h3 {{ color: #1e3d33 !important; }} .stButton>button, .stDownloadButton>button {{ border-radius: 12px !important; background-color: #2a5c4d !important; color: white !important; width: 100%; }} .stChatMessage {{ background-color: rgba(255, 255, 255, 0.85) !important; border-radius: 15px !important; backdrop-filter: blur(8px); }}</style>", unsafe_allow_html=True)

st.title("🧬 Robô Professor de Ciências")
st.markdown("---")

# CORREÇÃO CRUCIAL: URLs limpas sem espaços no início ou no fim
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
@st.cache_resource
def inicializar_sistema_completo():
    chave_api = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not chave_api:
        st.error("⚠️ Chave GOOGLE_API_KEY não configurada nos Secrets!")
        st.stop()
    return genai.Client(api_key=chave_api)

ai_client = inicializar_sistema_completo()

system_prompt = (
    "Você é um robô professor de ciências didático e divertido. "
    "Responda a qualquer dúvida ou conceito de ciências do aluno de forma clara. "
    "Foque estritamente em responder à pergunta de maneira educativa."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# 🧼 BARRA LATERAL METODOLOGIA DOWNLOAD
# ==============================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2a5c4d;'>📌 Painel do Aluno</h2>", unsafe_allow_html=True)
    
    # Links das Aulas como Botões de Acesso Direto limpos de espaços
    st.markdown("### 🎥 Assistir Aulas no Canal")
    for i, aula in enumerate(AULAS_DO_CANAL):
        st.link_button(label=f"▶️ {aula['titulo']}", url=aula['link'].strip(), key=f"link_aula_{i}")

    st.markdown("---")
    st.markdown("### 📚 Materiais de Apoio")
    for i, caminho in enumerate(["materiais/Ensino_Baseado_Simulacao.pdf", "materiais/Particulas.pdf", "materiais/Teoria_acido_base_Lewis.pdf"]):
        if os.path.exists(caminho):
            with open(caminho, "rb") as file:
                st.download_button(label=f"📥 Baixar Material {i+1}", data=file, file_name=os.path.basename(caminho), mime="application/pdf", key=f"mat_{i}")

    st.markdown("---")
    if st.button("🗑️ Limpar Conversa", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

# Mensagem de boas-vindas estática
if not st.session_state.messages:
    st.info("👋 **Olá, cientista!** Utilize a barra lateral para acessar as aulas e materiais ou digite sua dúvida sobre qualquer assunto de Ciências abaixo!")

# Histórico de conversas
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            pdf_data = gerar_pdf_resposta(st.session_state.messages[i-1]["content"], message["content"])
            st.download_button(label="📥 Baixar Resposta em PDF", data=pdf_data, file_name=f"resposta_{i}.pdf", mime="application/pdf", key=f"dl_{i}")

# Input de chat livre
prompt_usuario = st.chat_input("Digite sua dúvida de ciências...")

if prompt_usuario:
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    pergunta_atual = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Analisando os elements... 🧪"):
            try:
                resposta = ai_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=pergunta_atual,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt
                    )
                )
                texto_resposta = resposta.text if resposta.text else "Não consegui formular uma explicação."
            except Exception as e:
                texto_resposta = f"Erro na chamada da API com o modelo estável: {str(e)}"
            
            st.markdown(texto_resposta)
            
            pdf_dados = gerar_pdf_resposta(pergunta_atual, texto_resposta)
            st.download_button(label="📥 Baixar Resposta em PDF", data=pdf_dados, file_name="resposta_ciencias.pdf", mime="application/pdf", key="dl_imediato")
            
    st.session_state.messages.append({"role": "assistant", "content": texto_resposta})
