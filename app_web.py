import os
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Configuração da página Web com título e ícone customizados
st.set_page_config(
    page_title="Robô Professor de Ciências", 
    page_icon="🧬",
    layout="centered"
)

# ==============================================================================
# 🎨 DESIGN PREMIUM: Adiciona o Degradê de Fundo e Estiliza os Balões do Chat
# ==============================================================================
st.markdown("""
    <style>
        /* 🌌 DEFINE O DEGRADÊ DE FUNDO DA TELA INTEIRA */
        .stApp {
            background: linear-gradient(135deg, #eef5f3 0%, #dbe7e4 100%) !important;
            background-attachment: fixed;
        }

        /* 📌 Se você preferir usar uma IMAGEM de fundo em vez de degradê, 
           apague as linhas da .stApp acima e use as linhas abaixo:
        .stApp {
            background-image: url("https://unsplash.com");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        */

        /* Estilização dos títulos */
        h1, h2, h3 {
            color: #1e3d33 !important;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-weight: 700;
        }

        /* Customização dos botões da barra lateral escura */
        .stButton>button {
            border-radius: 12px !important;
            background-color: #2a5c4d !important;
            color: white !important;
            border: none !important;
            width: 100%;
            padding: 10px !important;
            font-weight: bold !important;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1e3d33 !important;
            transform: translateY(-2px);
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.1);
        }

        /* Deixa os balões de conversa levemente transparentes e elegantes */
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.75) !important;
            border-radius: 15px !important;
            padding: 15px !important;
            margin-bottom: 10px !important;
            box-shadow: 0px 2px 5px rgba(0,0,0,0.02) !important;
            backdrop-filter: blur(5px);
        }
    </style>
""", unsafe_allow_html=True)

# Cabeçalho Principal Estilizado
st.title("🧬 Robô Professor de Ciências")
st.markdown("---")


# ==============================================================================
# 🧠 CADASTRE SUAS AULAS AQUI (Texto resumido + Link limpo + Título + Sugestão)
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

# Variável auxiliar para capturar cliques nos botões de atalho
pergunta_clicada = None

# ==============================================================================
# 🧼 BARRA LATERAL: BOTÕES DE TEXTO, PDFS E LIMPEZA
# ==============================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2a5c4d;'>📌 Painel do Aluno</h2>", unsafe_allow_html=True)
    
    # NOVIDADE: Lista de aulas com botões de atalho para enviar a pergunta direto
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

# Mensagem inicial de boas-vindas
if len(st.session_state.messages) == 0:
    st.info("👋 **Olá, cientista!** Escolha uma das aulas na barra lateral ou digite sua dúvida aqui embaixo. Eu vou te explicar o assunto e carregar o vídeo correspondente!")

# Exibe as mensagens armazenadas usando avatares customizados
for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Captura a entrada: seja digitada na caixa ou clicada no atalho lateral
caixa_entrada = st.chat_input("Digite sua dúvida aqui...")
pergunta = caixa_entrada if caixa_entrada else pergunta_clicada

if pergunta:
    # Mensagem do Aluno
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

    # Resposta do Professor
    with st.chat_message("assistant", avatar="🤖"):
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
                st.markdown(f"🔗 **[Clique aqui para abrir diretamente no YouTube]({url_para_abrir})**")
                st.video(url_para_abrir)
                st.success("🎬 Vídeo da aula carregado acima com sucesso!")
                
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})
            
        except Exception as e:
            st.error(f"Erro ao acionar a IA: {e}")
            
    # Se a pergunta veio de um botão lateral, força o recarregamento para sincronizar o chat visualmente
    if pergunta_clicada:
        st.rerun()
