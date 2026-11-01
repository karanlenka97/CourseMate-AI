from __future__ import annotations
import hashlib
import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DB_DIR = Path("chroma_db")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

st.set_page_config(page_title="CourseMate.Ai", page_icon="◆", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
    :root { --bg:#08090b; --panel:#101114; --line:#22242a; --text:#f1f3f5; --muted:#858995; --green:#15c991; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--text); }
    .stApp { background:var(--bg); }
    [data-testid="stHeader"] { background:var(--bg); }
    [data-testid="stSidebar"] { background:#101114; border-right:1px solid #1c1d21; }
    [data-testid="stSidebar"] > div:first-child { padding-top:.8rem; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.55rem; }
    .brand { color:#f5f6f7; font-size:1.08rem; font-weight:700; padding:.7rem 0 .65rem; letter-spacing:-.035em; }
    .brand-mark { color:var(--green); margin-right:9px; font-size:1.18rem; }
    .micro-label { color:#838997; font:500 .68rem 'DM Mono',monospace; letter-spacing:.06em; text-transform:uppercase; }
    .engine-row { display:flex; justify-content:space-between; align-items:center; margin-top:.15rem; }
    .online { color:var(--green); font:500 .67rem 'DM Mono',monospace; }
    .model-name { font-size:.85rem; font-weight:600; margin-top:.35rem; }
    .sidebar-rule { height:1px; background:#202126; margin:.6rem -1rem .25rem; }
    .section-label { color:#838997; font:500 .68rem 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; margin:.7rem 0 .15rem; }
    .doc-card { background:#15161a; border:1px solid #23252b; border-radius:9px; padding:11px 12px; margin:.38rem 0; }
    .doc-name { color:#e7e9ed; font-size:.78rem; font-weight:600; overflow-wrap:anywhere; }
    .doc-meta { color:#747986; font:.68rem 'DM Mono',monospace; margin-top:5px; }
    .doc-dot { color:var(--green); }
    [data-testid="stFileUploader"] { border:1px dashed #34363e; border-radius:12px; background:#101114; padding:.45rem; }
    [data-testid="stFileUploader"] section { min-height:112px; }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color:#898d98 !important; }
    .stButton > button { color:#e8eaed; background:#15161a; border:1px solid #30323a; border-radius:8px; min-height:38px; font-size:.78rem; }
    .stButton > button:hover { border-color:var(--green); color:white; }
    [data-testid="stChatInput"] { position:fixed; bottom:1.6rem; left:calc(312px + (100vw - 312px - 64px) / 2); transform:translateX(-50%); width:min(880px,calc(100vw - 380px)) !important; background:#15161a; border:1px solid #25272e; border-radius:15px; }
    [data-testid="stChatInput"] textarea { color:#f1f3f5 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color:#7b7e87; }
    [data-testid="stChatInput"] button { background:#f4f5f6 !important; color:#101114 !important; border-radius:10px !important; }
    [data-testid="stChatMessage"] { background:transparent; border:0; padding:.45rem 0; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color:#e8eaf0; line-height:1.75; }
    .constraint { display:table; margin:1.25rem auto 2rem; color:#777a83; border:1px solid #24262c; border-radius:999px; padding:7px 14px; font:.68rem 'DM Mono',monospace; }
    .question-bubble { margin:0 0 1.7rem auto; max-width:780px; background:#17181c; border:1px solid #292b31; border-radius:15px; padding:15px 19px; color:#f2f3f6; text-align:right; line-height:1.55; }
    .answer-wrap { max-width:860px; margin:0 auto; }
    .answer-mark { display:inline-flex; align-items:center; justify-content:center; width:32px; height:32px; color:var(--green); background:#10221e; border:1px solid #183b32; border-radius:9px; margin-right:12px; }
    .source-chip { display:inline-block; background:#17181c; border:1px solid #292b31; border-radius:6px; color:#aab0bd; padding:5px 9px; margin:4px 5px 4px 0; font:.65rem 'DM Mono',monospace; }
    .composer-spacer { height:120px; }
    .powered { position:fixed; bottom:.45rem; left:calc(312px + (100vw - 312px) / 2); transform:translateX(-50%); color:#565a65; font:.58rem 'DM Mono',monospace; }
    .sidebar-footer { position:fixed; bottom:0; left:0; width:312px; border-top:1px solid #202126; background:#101114; padding:18px 20px; }
    [data-testid="stCaptionContainer"] { color:#818590; }
    @media (max-width: 800px) { [data-testid="stChatInput"] { left:50%; width:calc(100vw - 2rem) !important; } .powered { left:50%; } .sidebar-footer { position:static; width:auto; } }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Preparing your document search…")
def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings()

@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=get_embeddings(),
    )


def read_pdf(upload) -> list[Document]:
    reader = PdfReader(upload)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": upload.name, "page": page_number},
                )
            )
    return pages


def index_upload(upload) -> tuple[str, int]:
    raw = upload.getvalue()
    file_id = hashlib.sha256(raw).hexdigest()
    store = get_vectorstore()
    found = store.get(where={"file_id": file_id}, include=[])
    if found.get("ids"):
        return file_id, len(found["ids"])
    upload.seek(0)
    pages = read_pdf(upload)
    if not pages:
        raise ValueError("No selectable text was found. This PDF may be a scanned image.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=160)
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["file_id"] = file_id
    store.add_documents(chunks)
    return file_id, len(chunks)


def list_indexed_files() -> list[dict]:
    result = get_vectorstore().get(include=["metadatas"])
    files: dict[str, dict] = {}
    for metadata in result.get("metadatas", []):
        if metadata:
            file_id = metadata.get("file_id", metadata.get("source", ""))
            files[file_id] = {"name": metadata.get("source", "PDF document"), "id": file_id}
    return list(files.values())


def answer_question(question: str) -> tuple[str, list[Document]]:
    store = get_vectorstore()
    docs = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    ).invoke(question)
    if not docs:
        return "I could not find the answer in the document.", []
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant. Use ONLY the provided context to answer. "
                'If the answer is absent, say exactly: "I could not find the answer in the document." '
                "Be clear and concise. Do not make up facts.",
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'document')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )
    # load_dotenv() reads the app owner's key from .env; end users never enter it.
    model = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    result = model.invoke(prompt.format_prompt(context=context, question=question).to_messages())
    content = result.content
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content), docs


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">⬢</span>CourseMate.Ai</div>', unsafe_allow_html=True)
    st.markdown('<div class="engine-row"><span class="micro-label">LLM engine</span><span class="online">ONLINE</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-name">Gemini · {html.escape(MODEL)}</div><div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Knowledge base</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Drag and drop or click to browse. Text-based PDFs.",
    )
    if uploads and st.button("＋  Add to knowledge base", type="primary", use_container_width=True):
        for upload in uploads:
            try:
                with st.spinner(f"Reading {upload.name}…"):
                    _, count = index_upload(upload)
                st.success(f"Added {upload.name} · {count} chunks")
            except Exception as exc:
                st.error(f"Could not process {upload.name}: {exc}")

    try:
        indexed_files = list_indexed_files()
        st.markdown(f'<div class="section-label">Indexed files (ChromaDB) <span style="float:right;color:#d0d2d7">{len(indexed_files)}</span></div>', unsafe_allow_html=True)
        if indexed_files:
            for item in indexed_files:
                st.markdown(
                    f'<div class="doc-card"><div class="doc-name">📕 &nbsp;{html.escape(item["name"])}</div>'
                    '<div class="doc-meta"><span class="doc-dot">●</span> &nbsp;Indexed</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Your indexed PDFs will appear here.")
    except Exception:
        indexed_files = []

    st.markdown('<div class="sidebar-footer"><span class="micro-label">⚙ &nbsp; Retriever config (MMR)</span></div>', unsafe_allow_html=True)

st.markdown('<div class="constraint">ⓘ &nbsp; System constraint: Answer based ONLY on provided context.</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not indexed_files:
    st.markdown(
        '<div style="text-align:center;margin:18vh auto 0;max-width:520px;color:#8a8d96">'
        '<div style="font-size:2rem;color:#18c991;margin-bottom:12px">✳</div>'
        '<div style="font-size:1.2rem;color:#eceef1;font-weight:600">Your course materials, ready to explore</div>'
        '<p style="line-height:1.7">Upload a PDF in the left panel, then ask questions and get answers grounded in your documents.</p></div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="question-bubble">{html.escape(message["content"])}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="answer-wrap"><span class="answer-mark">✣</span>', unsafe_allow_html=True)
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown('<div class="section-label">⌕ &nbsp; Retrieved context</div>', unsafe_allow_html=True)
            chips = "".join(
                f'<span class="source-chip">📕 &nbsp;{html.escape(source["name"][:24])} · Pg {html.escape(str(source["page"]))}</span>'
                for source in message["sources"]
            )
            st.markdown(chips + "</div>", unsafe_allow_html=True)

question = st.chat_input("Ask a question based on uploaded context…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    if not indexed_files:
        st.warning("Upload a PDF and add it to the knowledge base before asking questions.")
    else:
        try:
            with st.spinner("Searching your documents…"):
                response, docs = answer_question(question)
            sources = []
            seen = set()
            for doc in docs:
                key = (doc.metadata.get("source", "document"), doc.metadata.get("page", "?"))
                if key not in seen:
                    seen.add(key)
                    sources.append({"name": key[0], "page": key[1]})
            st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
            st.rerun()
        except Exception as exc:
            error_text = str(exc)
            if "API_KEY" in error_text.upper() or "API KEY" in error_text.upper():
                st.error("Gemini credentials are not configured for this app. The app owner should set GOOGLE_API_KEY in the app's .env file.")
            else:
                st.error(f"I couldn't answer that just now. Check the Gemini connection and try again. Details: {exc}")

st.markdown('<div class="composer-spacer"></div><div class="powered">Powered by Google Generative AI (Gemini) &amp; LangChain</div>', unsafe_allow_html=True)


