import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference


load_dotenv()

APP_TITLE = "Technical Documentation Q&A"
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

RAGAS_METRICS = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithoutReference()]

METRIC_META = {
    "faithfulness": (
        "Faithfulness",
        "Is the answer grounded in the retrieved context? 1.0 means no hallucination.",
    ),
    "answer_relevancy": (
        "Answer Relevancy",
        "Does the answer actually address the question? 1.0 means fully on-topic.",
    ),
    "llm_context_precision_without_reference": (
        "Context Precision",
        "Are the retrieved chunks relevant to the question? 1.0 means all chunks were useful.",
    ),
}


def file_digest(file_payloads: tuple[tuple[str, bytes], ...]) -> str:
    hasher = hashlib.sha256()
    for file_name, file_bytes in file_payloads:
        hasher.update(file_name.encode("utf-8"))
        hasher.update(file_bytes)
    return hasher.hexdigest()


def load_documents_from_uploads(file_payloads: tuple[tuple[str, bytes], ...]) -> list[Document]:
    documents: list[Document] = []
    for file_name, file_bytes in file_payloads:
        suffix = Path(file_name).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        try:
            pages = PyPDFLoader(tmp_path).load()
            for page in pages:
                page.metadata["source_file"] = file_name
            documents.extend(pages)
        finally:
            os.unlink(tmp_path)
    return documents


@st.cache_resource(show_spinner=False)
def build_vector_store(file_key: str, file_payloads: tuple[tuple[str, bytes], ...]) -> FAISS:
    docs = load_documents_from_uploads(file_payloads)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def build_conversational_rag_chain(retriever):
    llm = ChatOpenAI(model=DEFAULT_CHAT_MODEL, temperature=0)

    # Reformulates follow-up questions into standalone queries so retrieval
    # works correctly even when the user references earlier context ("what about that?").
    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and the latest user question, reformulate the question "
            "into a standalone question understandable without the history. "
            "Return it as-is if no reformulation is needed. Do NOT answer it.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a technical documentation assistant. Answer questions using only "
            "the provided documentation context. If the answer is not covered, say so clearly. "
            "Be precise, use technical language appropriate to the docs, and cite specific "
            "sections or page numbers when the metadata is available.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def run_ragas_evaluation(question: str, answer: str, source_docs: list[Document]) -> dict[str, float]:
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=[doc.page_content for doc in source_docs],
    )
    result = evaluate(dataset=EvaluationDataset(samples=[sample]), metrics=RAGAS_METRICS)
    df = result.to_pandas()
    return {col: float(df[col].iloc[0]) for col in df.columns if col in METRIC_META}


def render_ragas_panel(scores: dict[str, float]) -> None:
    st.markdown("**RAGAS Evaluation**")
    cols = st.columns(len(METRIC_META))
    for col, (key, (label, help_text)) in zip(cols, METRIC_META.items()):
        if key in scores:
            score = scores[key]
            delta_color = "normal" if score >= 0.7 else "inverse"
            col.metric(label=label, value=f"{score:.2f}", help=help_text, delta_color=delta_color)


def render_source_chunks(source_docs: list[Document]) -> None:
    with st.expander(f"Retrieved {len(source_docs)} source chunks"):
        for idx, doc in enumerate(source_docs, start=1):
            source_name = doc.metadata.get("source_file", "Uploaded PDF")
            page_number = doc.metadata.get("page")
            page_label = f"Page {page_number + 1}" if isinstance(page_number, int) else "Page unknown"
            st.caption(f"**Chunk {idx}** · {source_name} · {page_label}")
            st.write(doc.page_content.strip())
            if idx < len(source_docs):
                st.divider()


def render_assistant_message(msg: dict, msg_idx: int) -> None:
    st.write(msg["content"])
    render_source_chunks(msg["sources"])

    eval_key = f"eval_{msg_idx}"
    if eval_key in st.session_state.evaluations:
        render_ragas_panel(st.session_state.evaluations[eval_key])
    elif st.button("Evaluate this answer", key=f"btn_{msg_idx}"):
        with st.spinner("Running RAGAS evaluation (this makes additional LLM calls)..."):
            try:
                scores = run_ragas_evaluation(msg["question"], msg["content"], msg["sources"])
                st.session_state.evaluations[eval_key] = scores
                st.rerun()
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📖", layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Upload technical documentation PDFs and ask questions. "
        "Click **Evaluate** on any answer to run RAGAS quality metrics."
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing `OPENAI_API_KEY`. Add it to `.env` before running.")
        st.stop()

    for key, default in (("messages", []), ("chat_history", []), ("evaluations", {})):
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.header("Setup")
        st.markdown(
            "- Upload API docs, framework manuals, or spec PDFs\n"
            "- Ask technical questions in the chat\n"
            "- Click **Evaluate** on any answer to score it with RAGAS"
        )
        st.caption(f"Chat model: `{DEFAULT_CHAT_MODEL}`")
        st.caption(f"Embedding model: `{DEFAULT_EMBEDDING_MODEL}`")
        st.divider()
        st.markdown("**RAGAS Metrics**")
        for _, (label, help_text) in METRIC_META.items():
            st.caption(f"**{label}**: {help_text}")
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.evaluations = {}
            st.rerun()

    uploaded_files = st.file_uploader(
        "Upload documentation PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one documentation PDF to start.")
        st.stop()

    file_payloads = tuple((f.name, f.getvalue()) for f in uploaded_files)
    current_key = file_digest(file_payloads)

    if st.session_state.get("file_key") != current_key:
        st.session_state.file_key = current_key
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.evaluations = {}

    with st.spinner("Indexing documentation..."):
        vector_store = build_vector_store(current_key, file_payloads)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    rag_chain = build_conversational_rag_chain(retriever)

    # Replay conversation history; track assistant message index for eval button keys
    assistant_idx = 0
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg, assistant_idx)
                assistant_idx += 1
            else:
                st.write(msg["content"])

    if question := st.chat_input("Ask a technical question about the documentation"):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                result = rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history,
                })
            answer = result["answer"]
            sources = result.get("context", [])
            new_msg = {"role": "assistant", "content": answer, "sources": sources, "question": question}
            render_assistant_message(new_msg, assistant_idx)

        st.session_state.messages.append(new_msg)
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer),
        ])


if __name__ == "__main__":
    main()
