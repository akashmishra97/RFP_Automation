import os
import io
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from llama_cloud_services import LlamaParse

from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.agent import FunctionAgent
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel
try:
    from RFP_Automation.app.workflow import run_workflow as run_workflow_mod
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[1])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from RFP_Automation.app.workflow import run_workflow as run_workflow_mod


# ---------------------------
# Config & Constants
# ---------------------------
load_dotenv()

AGENT_SYSTEM_PROMPT = """
You are a research agent tasked with filling out a specific form key/question with the appropriate value, given a bank of context.
You are given a specific form key/question. Think step-by-step and use the existing set of tools to help answer the question.

You MUST always use at least one tool to answer each question. Only after you've determined that existing tools do not \
answer the question should you try to reason from first principles and prior knowledge to answer the question.

You MUST try to answer the question instead of only saying 'I dont know'.
"""

EXTRACT_KEYS_PROMPT = """
You are provided an entire RFP document, or a large subsection from it.

We wish to generate a response to the RFP in a way that adheres to the instructions within the RFP, \
including the specific sections that an RFP response should contain, and the content that would need to go \
into each section.

Your task is to extract out a list of "questions", where each question corresponds to a specific section that is required in the RFP response.
Put another way, after we extract out the questions we will go through each question and answer each one \
with our downstream research assistant, and the combined
question:answer pairs will constitute the full RFP response.

You must TRY to extract out questions that can be answered by the provided knowledge base. We provide the list of file metadata below.

Additional requirements:
- Try to make the questions SPECIFIC given your knowledge of the RFP and the knowledge base. Instead of asking a question like \
"How do we ensure security" ask a question that actually addresses a security requirement in the RFP and can be addressed by the knowledge base.
- Make sure the questions are comprehensive and addresses all the RFP requirements.
- Make sure each question is descriptive - this gives our downstream assistant context to fill out the value for that question
- Extract out all the questions as a list of strings.


Knowledge Base Files:
{file_metadata}

RFP Full Template:
{rfp_text}
"""

GENERATE_OUTPUT_PROMPT = """
You are an expert analyst.
Your task is to generate an RFP response according to the given RFP and question/answer pairs.

You are given the following RFP and qa pairs:

<rfp_document>
{output_template}
</rfp_document>

<question_answer_pairs>
{answers}
</question_answer_pairs>

Not every question has an appropriate answer. This is because the agent tasked with answering the question did not have the right context to answer it.
If this is the case, you MUST come up with an answer that is reasonable. You CANNOT say that you are unsure in any area of the RFP response.


Please generate the output according to the template and the answers, in markdown format.
Directly output the generated markdown content, do not add any additional text, such as "```markdown" or "Here is the output:".
Follow the original format of the template as closely as possible, and fill in the answers into the appropriate sections.
"""

# Generate a single section for one question/answer pair, formatted per RFP style
GENERATE_SECTION_PROMPT = """
You are an expert analyst.
Given one RFP question and its answer, write the corresponding markdown section content that fits naturally into the RFP response.

Constraints:
- Use concise, professional tone.
- Follow typical RFP structure (heading then content). If the question itself is long, abbreviate it into a short heading.
- Do not include unrelated template sections.
- Keep the section self-contained and avoid repeating prior sections.

<question>
{question}
</question>

<answer>
{answer}
</answer>

Output only markdown for this section.
"""

class OutputQuestions(BaseModel):
    questions: List[str]


def ensure_dirs(base_out: Path) -> None:
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "workflow_output").mkdir(parents=True, exist_ok=True)


def build_retrieval_tools(index: VectorStoreIndex, files: List[str], summaries: Dict[str, str]) -> List[FunctionTool]:
    def generate_tool(file: str, file_description: Optional[str] = None) -> FunctionTool:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="file_path", operator=FilterOperator.EQ, value=file),
            ]
        )

        def chunk_retriever_fn(query: str) -> str:
            retriever = index.as_retriever(similarity_top_k=5, filters=filters)
            nodes = retriever.retrieve(query)
            full_text = "\n\n========================\n\n".join(
                [n.get_content(metadata_mode="all") for n in nodes]
            )
            return full_text

        fn_name = Path(file).stem + "_retrieve"
        tool_description = f"Retrieves a small set of relevant document chunks from {file}."
        if file_description:
            tool_description += f"\n\nFile Description: {file_description}"
        return FunctionTool.from_defaults(
            fn=chunk_retriever_fn, name=fn_name, description=tool_description
        )

    tools: List[FunctionTool] = []
    for f in files:
        tools.append(generate_tool(f, summaries.get(f)))
    return tools


async def run_workflow_legacy(
    rfp_pdf_path: str,
    context_pdf_paths: List[str],
    work_dir: Path,
    progress_cb: Optional[Any] = None,
    requests_per_minute: int = 8,
    max_questions_this_run: Optional[int] = None,
    resume_from_partial: bool = True,
) -> Dict[str, Any]:
    ensure_dirs(work_dir)
    wf_dir = work_dir / "workflow_output"

    # Models
    llm: LLM = Gemini(model="gemini-2.5-flash", timeout=600, max_output_tokens=8192)
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    parser = LlamaParse(
        parse_mode="parse_page_with_agent",
        model="gemini-2.5-flash",
        high_res_ocr=True,
        outlined_table_extraction=True,
        output_tables_as_HTML=True,
    )

    # Parse context documents with retries (handles transient DNS/network issues)
    async def _aparser_with_retry(inputs: Any, max_attempts: int = 4, base_delay: int = 5):
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await parser.aparse(inputs)
            except Exception as e:  # network/DNS/transient errors
                last_exc = e
                await asyncio.sleep(base_delay * attempt)
        assert last_exc is not None
        raise last_exc

    # Parse each context file one-by-one so we can show precise progress
    if progress_cb:
        progress_cb({"phase": "parse_context_start", "total": len(context_pdf_paths), "current": 0})
    file_dicts: Dict[str, Dict[str, Any]] = {}
    for idx, file_name in enumerate(context_pdf_paths):
        try:
            results_any = await _aparser_with_retry([file_name])
        except Exception as e:
            # Skip file on persistent parse failure
            if progress_cb:
                progress_cb({"phase": "parse_context_progress", "file": file_name, "total": len(context_pdf_paths), "current": idx + 1})
            continue

        # Normalize result (list vs single)
        result_obj = None
        if isinstance(results_any, list):
            if len(results_any) > 0:
                result_obj = results_any[0]
        else:
            result_obj = results_any

        if result_obj is None:
            # Nothing parsed for this file; skip
            if progress_cb:
                progress_cb({"phase": "parse_context_progress", "file": file_name, "total": len(context_pdf_paths), "current": idx + 1})
            continue

        docs = []
        try:
            docs = result_obj.get_markdown_nodes(split_by_page=True)
        except Exception:
            docs = []

        # Skip files that produced no text
        if not docs:
            if progress_cb:
                progress_cb({"phase": "parse_context_progress", "file": file_name, "total": len(context_pdf_paths), "current": idx + 1})
            continue

        file_dicts[file_name] = {"file_path": file_name, "docs": docs}
        if progress_cb:
            progress_cb({"phase": "parse_context_progress", "file": file_name, "total": len(context_pdf_paths), "current": idx + 1})

    # Summaries
    if progress_cb:
        progress_cb({"phase": "summarize_start", "total": len(context_pdf_paths), "current": 0})
    valid_files = [f for f in context_pdf_paths if f in file_dicts]
    for sidx, fpath in enumerate(valid_files):
        try:
            sindex = SummaryIndex(file_dicts[fpath]["docs"])
            resp = sindex.as_query_engine(llm=llm).query(
                "Generate a short 1-2 line summary of this file to help inform an agent on what this file is about."
            )
            file_dicts[fpath]["summary"] = str(resp)
        except Exception:
            file_dicts[fpath]["summary"] = ""
        if progress_cb:
            progress_cb({"phase": "summarize_progress", "file": fpath, "total": len(valid_files), "current": sidx + 1})

    # Build vector store
    persist_dir = str(work_dir / "storage_chroma_rfp")
    vector_store = ChromaVectorStore.from_params(collection_name="rfp_docs", persist_dir=persist_dir)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # Insert nodes (first-run)
    all_nodes: List[TextNode] = [c for d in file_dicts.values() for c in d["docs"]]
    if all_nodes:
        index.insert_nodes(all_nodes)
    if progress_cb:
        progress_cb({"phase": "index_built", "nodes": len(all_nodes)})

    # Tools
    summaries = {k: v.get("summary", "") for k, v in file_dicts.items()}
    tools = build_retrieval_tools(index, list(file_dicts.keys()), summaries)

    # Parse RFP template
    if progress_cb:
        progress_cb({"phase": "parse_rfp_start"})
    rfp_result = await _aparser_with_retry(rfp_pdf_path)
    rfp_docs = rfp_result.get_markdown_nodes(split_by_page=True)
    if progress_cb:
        progress_cb({"phase": "parse_rfp_done", "pages": len(rfp_docs)})

    # Extract questions
    all_text = "\n\n".join([d.get_content(metadata_mode="all") for d in rfp_docs])
    prompt = PromptTemplate(template=EXTRACT_KEYS_PROMPT)
    file_metadata = "\n\n".join([
        f"Name:{t.metadata.name}\nDescription:{t.metadata.description}" for t in tools
    ])
    output_qs = (
        await llm.astructured_predict(
            OutputQuestions,
            prompt,
            file_metadata=file_metadata,
            rfp_text=all_text,
        )
    ).questions
    if progress_cb:
        progress_cb({"phase": "questions_extracted", "total": len(output_qs)})
    # Persist questions immediately so they exist even if later steps fail
    try:
        with open(wf_dir / "all_keys.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_qs))
    except Exception:
        pass

    # Answer questions
    answers: List[Dict[str, str]] = []
    research_agent = FunctionAgent(tools=tools, llm=llm, system_prompt=AGENT_SYSTEM_PROMPT)
    # Prepare partial answers file (truncate/initialize)
    try:
        with open(wf_dir / "combined_answers.jsonl", "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
    # If resuming, load already-answered questions and skip them
    answered_set = set()
    if resume_from_partial and (wf_dir / "combined_answers.jsonl").exists():
        try:
            with open(wf_dir / "combined_answers.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "question" in obj:
                            answered_set.add(obj["question"])
                    except Exception:
                        continue
        except Exception:
            pass

    remaining_qs: List[str] = [q for q in output_qs if q not in answered_set]
    if max_questions_this_run is not None and max_questions_this_run > 0:
        remaining_qs = remaining_qs[:max_questions_this_run]

    # Simple RPM pacing
    min_gap_sec = max(0.0, 60.0 / max(1, requests_per_minute))
    last_call_ts: Optional[float] = None

    for qidx, q in enumerate(remaining_qs):
        # pace requests to avoid 429
        if last_call_ts is not None:
            import time
            gap = time.time() - last_call_ts
            if gap < min_gap_sec:
                await asyncio.sleep(min_gap_sec - gap)

        # try with basic backoff on 429
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await research_agent.run(q)
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    # parse suggested retry seconds if present
                    import re
                    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}", msg)
                    wait_s = int(m.group(1)) if m else 35
                    if progress_cb:
                        progress_cb({"phase": "rate_limited", "wait_seconds": wait_s})
                    await asyncio.sleep(wait_s)
                    if attempt < 4:
                        continue
                # Other errors or exhausted retries: record error answer and continue
                qa_err = {"question": q, "answer": "", "error": msg}
                answers.append(qa_err)
                try:
                    with open(wf_dir / "combined_answers.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(qa_err) + "\n")
                except Exception:
                    pass
                if progress_cb:
                    progress_cb({"phase": "qa_progress", "current": qidx + 1, "total": len(remaining_qs)})
                resp = None
                break

        # record call time
        import time
        last_call_ts = time.time()

        if resp is not None:
            qa = {"question": q, "answer": str(resp)}
            answers.append(qa)
            # Append after each answer so partial progress is saved
            try:
                with open(wf_dir / "combined_answers.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(qa) + "\n")
            except Exception:
                pass
        if progress_cb:
            progress_cb({"phase": "qa_progress", "current": qidx + 1, "total": len(remaining_qs)})

    combined_answers = "\n".join([json.dumps(a) for a in answers])

    # Generate output in small sections to avoid token limits
    q_to_a: Dict[str, str] = {a.get("question", ""): a.get("answer", "") for a in answers}
    section_md_list: List[str] = []
    sec_prompt = PromptTemplate(GENERATE_SECTION_PROMPT)
    if progress_cb:
        progress_cb({"phase": "generating_sections_start", "total": len(output_qs)})
    for sidx, q in enumerate(output_qs):
        a_text = q_to_a.get(q, "")
        if not a_text:
            sec_md = f"\n\n### {q[:80]}\n_TBD: No answer available._\n"
        else:
            try:
                sec_stream = await llm.astream(sec_prompt, question=q, answer=a_text)
                sec_md = ""
                async for chunk in sec_stream:
                    sec_md += chunk
            except Exception:
                sec_md = f"\n\n### {q[:80]}\n{a_text}\n"
        section_md_list.append(sec_md)
        if progress_cb:
            progress_cb({"phase": "generating_sections_progress", "current": sidx + 1, "total": len(output_qs)})

    final_output = "\n\n".join(section_md_list)

    # Save outputs
    with open(wf_dir / "final_output.md", "w", encoding="utf-8") as f:
        f.write(final_output)

    return {
        "questions": output_qs,
        "answers": answers,
        "markdown": final_output,
        "output_dir": str(wf_dir),
    }


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="RFP Generator (Gemini 2.5)", layout="wide")
st.title("RFP Response Generator (Gemini 2.5)")

col1, col2 = st.columns(2)
with col1:
    rfp_file = st.file_uploader("Upload RFP template PDF", type=["pdf"])
with col2:
    context_files = st.file_uploader(
        "Upload context PDFs (multiple)", type=["pdf"], accept_multiple_files=True
    )

work_dir = Path("data_out_rfp_ui")

progress = st.empty()
phase_text = st.empty()

if st.button("Generate Response", type="primary"):
    if not rfp_file:
        st.error("Please upload the RFP template PDF.")
        st.stop()
    if not context_files:
        st.error("Please upload at least one context PDF.")
        st.stop()

    # Save uploaded files to disk
    data_dir = Path("data_ui")
    data_dir.mkdir(parents=True, exist_ok=True)

    rfp_path = str(data_dir / rfp_file.name)
    with open(rfp_path, "wb") as f:
        f.write(rfp_file.read())

    context_paths: List[str] = []
    for cf in context_files:
        file_path = str(data_dir / cf.name)
        with open(file_path, "wb") as f:
            f.write(cf.read())
        context_paths.append(file_path)

    with st.status("Running workflow. This may take several minutes...", expanded=True) as status:
        st.write("Parsing documents and building index...")

        def cb(update: Dict[str, Any]):
            ph = update.get("phase", "")
            if ph == "parse_context_start":
                phase_text.info(f"Parsing context PDFs (0/{update['total']})")
                progress.progress(0.0)
            elif ph == "parse_context_progress":
                phase_text.info(f"Parsing context PDFs ({update['current']}/{update['total']}) - {Path(update['file']).name}")
                progress.progress(update["current"] / max(1, update["total"]))
            elif ph == "summarize_start":
                phase_text.info(f"Summarizing context (0/{update['total']})")
                progress.progress(0.0)
            elif ph == "summarize_progress":
                phase_text.info(f"Summarizing context ({update['current']}/{update['total']}) - {Path(update['file']).name}")
                progress.progress(update["current"] / max(1, update["total"]))
            elif ph == "index_built":
                phase_text.success("Index built")
                progress.progress(1.0)
            elif ph == "parse_rfp_start":
                phase_text.info("Parsing RFP template...")
                progress.progress(0.0)
            elif ph == "parse_rfp_done":
                phase_text.success(f"RFP parsed: {update.get('pages', '?')} pages")
                progress.progress(1.0)
            elif ph == "questions_extracted":
                phase_text.info(f"Extracted {update['total']} questions")
                progress.progress(1.0)
            elif ph == "qa_progress":
                phase_text.info(f"Answering questions ({update['current']}/{update['total']})")
                progress.progress(update["current"] / max(1, update["total"]))
            elif ph == "generating_output":
                phase_text.info(f"Generating output... {update['chars']} chars")

        try:
            # Keep RPM under free-tier limit and optionally cap questions per run
            result = asyncio.run(
                run_workflow_mod(
                    rfp_path,
                    context_paths,
                    work_dir,
                    progress_cb=cb,
                    requests_per_minute=8,
                    max_questions_this_run=None,
                    resume_from_partial=True,
                )
            )
        except Exception as e:
            st.error(f"Workflow failed: {e}")
            status.update(label="Failed", state="error")
            st.stop()
        status.update(label="Completed", state="complete")

    st.subheader("Generated Response")
    st.markdown(result["markdown"])

    st.download_button(
        label="Download Markdown",
        data=result["markdown"].encode("utf-8"),
        file_name="rfp_response.md",
        mime="text/markdown",
    )

    st.caption(f"Outputs saved to: {result['output_dir']}")


