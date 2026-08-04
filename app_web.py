import os
import json
import requests
import streamlit as st
import numpy as np
import base64
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Configuração da página Web básica e centralizada
st.set_page_config(page_title="Robô Professor de Ciências", page_icon="🧬", layout="centered")

# Credenciais de Integração com o GitHub API
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "") 
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")

# Pastas padrões de armazenamento local
PASTA_MATERIAIS = "materiais"
PASTA_VIDEOS = "videos"

os.makedirs(PASTA_MATERIAIS, exist_ok=True)
os.makedirs(PASTA_VIDEOS, exist_ok=True)

# 📄 ESTRUTURAÇÃO DO PDF DA RESPOSTA
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
# 🛠️ FUNÇÕES DE SINCRONIZAÇÃO AUTOMÁTICA COM O GITHUB VIA API (CORRIGIDA)
# ------------------------------------------------------------------------------
def enviar_arquivo_github(caminho_repositorio, conteudo_bytes, mensagem_commit):
    url = f"https://github.com{GITHUB_REPO}/contents/{caminho_repositorio}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}", 
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        r = requests.get(url, headers=headers)
        sha = r.json().get("sha") if r.status_code == 200 else None
        conteudo_base64 = base64.b64encode(conteudo_bytes).decode("utf-8")
        dados = {"message": mensagem_commit, "content": conteudo_base64}
        if sha:
            dados["sha"] = sha
        res = requests.put(url, headers=headers, json=dados)
        # CORREÇÃO CRÍTICA: Linha restaurada com os códigos de sucesso corretos
        return res.status_code in [200, 201]
    except Exception as e:
        st.error(f"Erro na conexão com o GitHub: {e}")
        return False

def deletar_arquivo_github(caminho_repositorio, mensagem_commit):
    url = f"https://github.com{GITHUB_REPO}/contents/{caminho_repositorio}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}", 
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            sha = r.json().get("sha")
            dados = {"message": mensagem_commit, "sha": sha}
            res = requests.delete(url, headers=headers, json=dados)
            return res.status_code == 200
        return False
    except Exception as e:
        st.error(f"Erro ao deletar no GitHub: {e}")
        return False
        
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

img_base64 = get_base64_image("fundo.jpg")
css_fundo = f'.stApp {{ background-image: url("data:image/jpg;base64,{img_base64}"); background-size: cover; background-attachment: fixed; }}' if img_base64 else '.stApp { background: linear-gradient(135deg, #eef5f3 0%, #dbe7e4 100%) !important; }'
st.markdown(f"<style>{css_fundo} h1, h2, h3 {{ color: #1e3d33 !important; }} .stButton>button, .stDownloadButton>button {{ border-radius: 12px !important; background-color: #2a5c4d !important; color: white !important; width: 100%; }}</style>", unsafe_allow_html=True)

st.title("🧬 Robô Professor de Ciências")
st.markdown("---")

@st.cache_resource
def inicializar_sistema_completo():
    chave_api = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not chave_api:
        st.error("⚠️ Chave GOOGLE_API_KEY não configurada nos Secrets!")
        st.stop()
    return genai.Client(api_key=chave_api)
ai_client = inicializar_sistema_completo()
system_prompt = "Você é um robô professor de ciências didático. Responda de forma clara, educativa e sempre em português do Brasil."

if "messages" not in st.session_state:
    st.session_state.messages = []

# Inicializa as listas na memória para garantir que não sumam após o upload
if "videos_memoria" not in st.session_state:
    st.session_state.videos_memoria = {}
if "pdfs_memoria" not in st.session_state:
    st.session_state.pdfs_memoria = {}

# Sincroniza os arquivos físicos existentes nas pastas (caso existam no GitHub)
try:
    for f in os.listdir(PASTA_VIDEOS):
        caminho = os.path.join(PASTA_VIDEOS, f)
        if f.endswith(('.mp4', '.mov', '.avi')) and os.path.getsize(caminho) > 0 and f not in st.session_state.videos_memoria:
            with open(caminho, "rb") as vf:
                st.session_state.videos_memoria[f] = vf.read()
                
    for f in os.listdir(PASTA_MATERIAIS):
        caminho = os.path.join(PASTA_MATERIAIS, f)
        if f.endswith('.pdf') and os.path.getsize(caminho) > 0 and f not in st.session_state.pdfs_memoria:
            with open(caminho, "rb") as pf:
                st.session_state.pdfs_memoria[f] = pf.read()
except Exception:
    pass

# ==============================================================================
# 🧼 BARRA LATERAL FIXA DO ALUNO + GERENCIADOR DO PROFESSOR (ADMIN)
# ==============================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2a5c4d;'>📌 Painel do Aluno</h2>", unsafe_allow_html=True)
    
    st.markdown("### 🎥 Assistir Aulas Gravadas")
    if st.session_state.videos_memoria:
        for i, (nome_video, video_bytes) in enumerate(st.session_state.videos_memoria.items()):
            st.markdown(f"**▶️ {nome_video}**")
            try:
                # CORREÇÃO: Transforma os bytes em uma URL string codificada em Base64
                # Isso força o navegador a abrir o vídeo nativamente, contornando o erro de codec do servidor
                base64_vid = base64.b64encode(video_bytes).decode("utf-8")
                video_url = f"data:video/mp4;base64,{base64_vid}"
                st.video(video_url, format="video/mp4", key=f"player_local_{i}")
            except Exception:
                st.error("Erro crítico ao renderizar o vídeo.")
                
            st.download_button(
                label=f"📥 Baixar Aula: {nome_video.replace('.mp4','')}", 
                data=video_bytes, 
                file_name=nome_video, 
                mime="video/mp4", 
                key=f"dl_vid_{i}"
            )
    else:
        st.info("Nenhum vídeo disponível.")

    st.markdown("---")
    st.markdown("### 📚 Materiais de Apoio")
    if st.session_state.pdfs_memoria:
        for i, (nome_arquivo, pdf_bytes) in enumerate(st.session_state.pdfs_memoria.items()):
            st.download_button(
                label=f"📥 Baixar {nome_arquivo.replace('.pdf', '')}", 
                data=pdf_bytes, 
                file_name=nome_arquivo, 
                mime="application/pdf", 
                key=f"mat_dinamico_{i}"
            )
    else:
        st.info("Nenhum material disponível.")

    st.markdown("---")
    if st.button("🗑️ Limpar Conversa", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    modo_admin = st.checkbox("⚙️ Acesso do Professor (Admin)", value=False)
    if modo_admin:
        senha = st.text_input("Senha mestra:", type="password", key="admin_password_field")
        if senha and senha == ADMIN_PASSWORD:
            st.success("🔒 Painel Liberado!")
            aba_pdf, aba_video = st.tabs(["📚 Apostilas (PDF)", "🎥 Vídeos (MP4)"])
            
            with aba_pdf:
                st.markdown("**Upload de Apostilas**")
                upload_pdf = st.file_uploader("Escolha o arquivo PDF:", type=["pdf"])
                if upload_pdf is not None and st.button("Salvar PDF"):
                    conteudo_pdf = upload_pdf.getvalue()
                    caminho_final_pdf = f"materiais/{upload_pdf.name}"
                    
                    st.session_state.pdfs_memoria[upload_pdf.name] = conteudo_pdf
                    enviar_arquivo_github(caminho_final_pdf, conteudo_pdf, f"Adicionando {upload_pdf.name}")
                    st.success("Salvo com sucesso!")
                    st.rerun()
                
                if st.session_state.pdfs_memoria:
                    arq_selecionado = st.selectbox("Apagar PDF:", list(st.session_state.pdfs_memoria.keys()))
                    if st.button("❌ Deletar Selecionado", type="primary"):
                        caminho_deletar_pdf = f"materiais/{arq_selecionado}"
                        deletar_arquivo_github(caminho_deletar_pdf, f"Deletando {arq_selecionado}")
                        if arq_selecionado in st.session_state.pdfs_memoria:
                            del st.session_state.pdfs_memoria[arq_selecionado]
                        try:
                            os.remove(os.path.join(PASTA_MATERIAIS, arq_selecionado))
                        except Exception:
                            pass
                        st.success("Apagado com sucesso!")
                        st.rerun()
            
            with aba_video:
                st.markdown("**Upload de Videoaulas**")
                upload_video = st.file_uploader("Escolha o arquivo de vídeo:", type=["mp4", "mov", "avi"])
                if upload_video is not None and st.button("Salvar Vídeo"):
                    conteudo_video = upload_video.getvalue()
                    caminho_final_video = f"videos/{upload_video.name}"
                    
                    st.session_state.videos_memoria[upload_video.name] = conteudo_video
                    enviar_arquivo_github(caminho_final_video, conteudo_video, f"Adicionando video {upload_video.name}")
                    st.success("Vídeo Salvo com sucesso!")
                    st.rerun()
                
                if st.session_state.videos_memoria:
                    vid_selecionado = st.selectbox("Apagar Vídeo:", list(st.session_state.videos_memoria.keys()))
                    if st.button("❌ Deletar Vídeo Selecionado", type="primary"):
                        caminho_deletar_video = f"videos/{vid_selecionado}"
                        deletar_arquivo_github(caminho_deletar_video, f"Deletando video {vid_selecionado}")
                        if vid_selecionado in st.session_state.videos_memoria:
                            del st.session_state.videos_memoria[vid_selecionado]
                        try:
                            os.remove(os.path.join(PASTA_VIDEOS, vid_selecionado))
                        except Exception:
                            pass
                        st.success("Vídeo Apagado com sucesso!")
                        st.rerun()

# ==============================================================================
# 💬 INTERFACE DE CHAT (ÁREA PRINCIPAL)
# ==============================================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "pdf_data" in message:
            st.download_button(
                label="📥 Baixar Resposta em PDF", 
                data=message["pdf_data"],
                file_name="resposta_ciencias.pdf", 
                mime="application/pdf", 
                key=f"dl_{message['id']}"
            )

if prompt := st.chat_input("Pergunte algo sobre ciências..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Respondendo..."):
            try:
                contents = []
                for msg in st.session_state.messages[:-1]:
                    contents.append(types.Content(
                        role="user" if msg["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=msg["content"])]
                    ))
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

                response = ai_client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=system_prompt),
                )
                resposta_texto = response.text
                message_placeholder.markdown(resposta_texto)
                
                pdf_buffer = gerar_pdf_resposta(prompt, resposta_texto)
                pdf_bytes = pdf_buffer.getvalue()
                
                msg_id = len(st.session_state.messages)
                st.download_button(
                    label="📥 Baixar Resposta em PDF", 
                    data=pdf_bytes,
                    file_name="resposta_ciencias.pdf", 
                    mime="application/pdf", 
                    key=f"dl_{msg_id}"
                )
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": resposta_texto, 
                    "pdf_data": pdf_bytes, 
                    "id": msg_id
                })
                
            except Exception as e:
                st.error(f"Erro ao processar resposta da IA: {e}")
