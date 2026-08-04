import os
import json
import requests
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

# Configuração da página Web fixa e centralizada
st.set_page_config(page_title="Robô Professor de Ciências", page_icon="🧬", layout="centered")

# Credenciais de Integração com o GitHub API
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")

# Pasta padrão de materiais de apoio
PASTA_MATERIAIS = "materiais"

# ------------------------------------------------------------------------------
# 🛠️ FUNÇÕES DE SINCRONIZAÇÃO AUTOMÁTICA COM O GITHUB VIA API
# ------------------------------------------------------------------------------
def enviar_arquivo_github(caminho_repositorio, conteudo_bytes, mensagem_commit):
    url = f"https://github.com{GITHUB_REPO}/contents/{caminho_repositorio}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    r = requests.get(url, headers=headers)
    sha = r.json().get("sha") if r.status_code == 200 else None
    conteudo_base64 = base64.b64encode(conteudo_bytes).decode("utf-8")
    dados = {"message": mensagem_commit, "content": conteudo_base64}
    if sha:
        dados["sha"] = sha
    res = requests.put(url, headers=headers, json=dados)
    return res.status_code in [200, 201]

def deletar_arquivo_github(caminho_repositorio, mensagem_commit):
    url = f"https://github.com{GITHUB_REPO}/contents/{caminho_repositorio}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        sha = r.json().get("sha")
        dados = {"message": mensagem_commit, "sha": sha}
        res = requests.delete(url, headers=headers, json=dados)
        return res.status_code == 200
    return False

# ------------------------------------------------------------------------------
# 📄 ESTRUTURAÇÃO DO PDF DA RESPOSTA
# ------------------------------------------------------------------------------
def gerar_pdf_resposta(pergunta, resposta):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1e3d33'), spaceAfter=12)
    estilo_pergunta = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#2a5c4d'), spaceAfter=12)
    estilo_corpo = ParagraphStyle('C1', parent=styles['Normal'], fontSize=11, leading=16, textColor=colors.HexColor('#333333'), spaceAfter=8)
    
    story = [Paragraph("🧬 Robô Professor de Ciências — Resposta", estilo_titulo), Spacer(1, 10), Paragraph(f"<b>Dúvida do Aluno:</b> {pergunta}", estilo_pergunta), Spacer(1, 10)]
    for linha in resposta.split('\n'):
        if linha.strip():
            story.append(Paragraph(linha.strip(), estilo_corpo))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ------------------------------------------------------------------------------
# 🖼️ ESTILIZAÇÃO E DESIGN DA INTERFACE (CSS)
# ------------------------------------------------------------------------------
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

# Carregamento da configuração externa de vídeos
JSON_PATH = "config_aulas.json"
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        AULAS_DO_CANAL = json.load(f)
else:
    AULAS_DO_CANAL = [
        {"titulo": "🌌 Aula: O que é química?", "link": "https://youtube.com"},
        {"titulo": "🌱 Aula: O que são partículas?", "link": "https://youtube.com"},
        {"titulo": "🪐 Aula: Soluções químicas", "link": "https://youtube.com"}
    ]

@st.cache_resource
def inicializar_sistema_completo():
    chave_api = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not chave_api:
        st.error("⚠️ Chave GOOGLE_API_KEY não configurada nos Secrets!")
        st.stop()
    return genai.Client(api_key=chave_api, http_options={'api_version': 'v1'})

ai_client = inicializar_sistema_completo()
system_prompt = "Você é um robô professor de ciências didático. Responda de forma clara e educativa."

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# 🧼 BARRA LATERAL FIXA DO ALUNO
# ==============================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2a5c4d;'>📌 Painel do Aluno</h2>", unsafe_allow_html=True)
    
    st.markdown("### 🎥 Assistir Aulas no Canal")
    for i, aula in enumerate(AULAS_DO_CANAL):
        st.link_button(label=f"▶️ {aula['titulo']}", url=aula['link'].strip(), key=f"link_aula_{i}")

    st.markdown("---")
    st.markdown("### 📚 Materiais de Apoio")
    if os.path.exists(PASTA_MATERIAIS):
        arquivos = [f for f in os.listdir(PASTA_MATERIAIS) if f.endswith('.pdf')]
        for i, nome_arquivo in enumerate(arquivos):
            with open(os.path.join(PASTA_MATERIAIS, nome_arquivo), "rb") as file:
                st.download_button(label=f"📥 Baixar {nome_arquivo.replace('.pdf', '')}", data=file, file_name=nome_arquivo, mime="application/pdf", key=f"mat_dinamico_{i}")

    st.markdown("---")
    if st.button("🗑️ Limpar Conversa", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# ⚙️ PAINEL DO PROFESSOR (DENTRO DE UM EXPANDER TOTALMENTE INDEPENDENTE)
# ==============================================================================
with st.expander("⚙️ Área do Professor (Painel de Controle)"):
    senha = st.text_input("Insira a senha mestra para gerenciar:", type="password", key="admin_password_field")
    if senha == ADMIN_PASSWORD:
        st.success("Acesso Autorizado!")
        
        st.markdown("### 📑 Adicionar ou Remover Apostilas (PDFs)")
        upload_pdf = st.file_uploader("Arraste uma nova apostila PDF aqui:", type=["pdf"])
        if upload_pdf is not None and st.button("Salvar Apostila no Sistema"):
            if enviar_arquivo_github(f"materiais/{upload_pdf.name}", upload_pdf.getvalue(), f"Adicionando {upload_pdf.name}"):
                st.success("Apostila salva com sucesso!")
                st.rerun()
                
        if os.path.exists(PASTA_MATERIAIS):
            arquivos_deletar = [f for f in os.listdir(PASTA_MATERIAIS) if f.endswith('.pdf')]
            if arquivos_deletar:
                arq_selecionado = st.selectbox("Selecione um material para apagar:", arquivos_deletar)
                if st.button("❌ EXCLUIR MATERIAL SELECIONADO", type="primary"):
                    if deletar_arquivo_github(f"materiais/{arq_selecionado}", f"Deletando {arq_selecionado}"):
                        st.success("Removido com sucesso!")
                        st.rerun()

        st.markdown("---")
        st.markdown("### 🎥 Adicionar ou Remover Vídeos")
        with st.form("nova_aula_form"):
            novo_titulo = st.text_input("Título da Aula")
            novo_link = st.text_input("Link do YouTube")
            submetido = st.form_submit_button("Cadastrar Vídeo")
            if submetido and novo_titulo and novo_link:
                AULAS_DO_CANAL.append({"titulo": novo_titulo, "link": novo_link})
                enviar_arquivo_github(JSON_PATH, json.dumps(AULAS_DO_CANAL, indent=4, ensure_ascii=False).encode("utf-8"), "Atualizando vídeos")
                st.success("Vídeo cadastrado!")
                st.rerun()
        
        if AULAS_DO_CANAL:
            st.write("Aulas Cadastradas:")
            for idx, item in enumerate(AULAS_DO_CANAL):
                col1, col2 = st.columns()
                col1.write(f"**{item['titulo']}**")
                if col2.button("Apagar", key=f"del_aula_{idx}"):
                    AULAS_DO_CANAL.pop(idx)
                    enviar_arquivo_github(JSON_PATH, json.dumps(AULAS_DO_CANAL, indent=4, ensure_ascii=False).encode("utf-8"), "Removendo vídeo")
                    st.rerun()
    elif senha != "":
        st.error("Senha incorreta!")

# ==============================================================================
# 💬 FLUXO PRINCIPAL DO CHAT DO ALUNO (ESTRUTURA INDEPENDENTE E FIXA)
# ==============================================================================
if not st.session_state.messages:
