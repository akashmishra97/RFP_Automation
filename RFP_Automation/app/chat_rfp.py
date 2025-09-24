import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

import streamlit as st
from dotenv import load_dotenv
import re

from llama_index.core.llms import LLM
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.fastembed import FastEmbedEmbedding

try:
    from RFP_Automation.app.parsing import build_parser, aparse_with_retry
    from RFP_Automation.app.indexing import build_index
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from RFP_Automation.app.parsing import build_parser, aparse_with_retry
    from RFP_Automation.app.indexing import build_index


load_dotenv()


def _save_uploaded_file(uploaded, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / uploaded.name
    with open(out_path, "wb") as f:
        f.write(uploaded.read())
    return str(out_path)


def _ensure_chat_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # [{"role": "user"|"assistant", "content": str}]
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "rfp_ready" not in st.session_state:
        st.session_state.rfp_ready = False


def _extract_first_html_table(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    try:
        m = re.search(r"<table[\s\S]*?</table>", text, re.IGNORECASE)
        return m.group(0) if m else None
    except Exception:
        return None


def build_engine_for_rfp_and_context(rfp_pdf_path: str, context_pdf_paths: List[str], work_dir: Path) -> Any:
    llm: LLM = Gemini(model="gemini-2.5-flash", timeout=600, max_output_tokens=4096)
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    parser = build_parser()

    # Parse RFP into nodes
    import asyncio
    r = asyncio.run(aparse_with_retry(parser, rfp_pdf_path))
    rfp_nodes = r.get_markdown_nodes(split_by_page=True)

    # Parse context PDFs
    all_nodes = list(rfp_nodes)
    for cpath in context_pdf_paths:
        try:
            c = asyncio.run(aparse_with_retry(parser, [cpath]))
            # If list, take first; else use value
            cobj = c[0] if isinstance(c, list) and c else (c if not isinstance(c, list) else None)
            if cobj is None:
                continue
            cnodes = cobj.get_markdown_nodes(split_by_page=True)
            all_nodes.extend(cnodes)
        except Exception:
            continue

    # Build a fresh index each run
    persist_dir_path = work_dir / "storage_chroma_rfp_chat"
    if persist_dir_path.exists():
        try:
            shutil.rmtree(persist_dir_path)
        except Exception:
            pass
    persist_dir = str(persist_dir_path)
    index = build_index(persist_dir, embed_model)
    if all_nodes:
        index.insert_nodes(all_nodes)

    # Build query engine over RFP only
    engine = index.as_query_engine(llm=llm, similarity_top_k=5)
    return engine


def main() -> None:
    st.set_page_config(page_title="RFP Chat", layout="wide")
    st.title("RFP Chatbot")
    st.caption("Ask questions about your uploaded RFP. The bot will cite the document.")

    _ensure_chat_state()

    with st.sidebar:
        st.header("RFP and Context")
        rfp_file = st.file_uploader("Upload RFP PDF", type=["pdf"])
        context_files = st.file_uploader("Upload context PDFs (optional)", type=["pdf"], accept_multiple_files=True)
        work_dir = Path("data_out_rfp_chat")

        if st.button("Process RFP", type="primary", disabled=rfp_file is None):
            if not rfp_file:
                st.error("Please upload an RFP PDF first.")
                st.stop()
            save_dir = Path("data_ui")
            rfp_path = _save_uploaded_file(rfp_file, save_dir)
            context_paths: List[str] = []
            if context_files:
                for cf in context_files:
                    try:
                        context_paths.append(_save_uploaded_file(cf, save_dir))
                    except Exception:
                        continue
            try:
                st.info("Building index. This can take a minute...")
                engine = build_engine_for_rfp_and_context(rfp_path, context_paths, work_dir)
                st.session_state.query_engine = engine
                st.session_state.rfp_ready = True
                st.success("RFP processed. You can start chatting.")
            except Exception as e:
                st.session_state.rfp_ready = False
                st.error(f"Failed to process RFP: {e}")

    if not st.session_state.rfp_ready:
        st.info("Upload and process an RFP PDF to begin.")
        return

    # Show chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about the RFP...")
    if user_q:
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            try:
                engine = st.session_state.query_engine
                resp = engine.query(user_q)
                ans = str(resp)
                st.markdown(ans)

                # Citations
                try:
                    source_nodes = getattr(resp, "source_nodes", []) or []
                    if source_nodes:
                        with st.expander("Sources"):
                            for idx, sn in enumerate(source_nodes, 1):
                                node_text = sn.get_content("none")
                                content_preview = (node_text or "")[:300].replace("\n", " ")
                                meta = sn.metadata or {}
                                page = meta.get("page") or meta.get("page_label") or "?"
                                st.markdown(f"**{idx}. Page {page}:** {content_preview}...")

                                # Render first HTML table if present
                                table_html = _extract_first_html_table(node_text or "")
                                if table_html:
                                    with st.expander(f"View table from source {idx}"):
                                        st.markdown(table_html, unsafe_allow_html=True)
                except Exception:
                    pass

                st.session_state.chat_messages.append({"role": "assistant", "content": ans})
            except Exception as e:
                err = f"Error answering: {e}"
                st.error(err)
                st.session_state.chat_messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()


