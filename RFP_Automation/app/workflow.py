import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.llms import LLM
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.agent import FunctionAgent
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core import SummaryIndex, VectorStoreIndex

from .parsing import build_parser, aparse_with_retry
from .indexing import build_index, generate_tools, summarize_file_nodes
from .prompts import EXTRACT_KEYS_PROMPT, GENERATE_SECTION_PROMPT
from pydantic import BaseModel
from .utils import ensure_dirs, append_jsonl


async def run_workflow(
    rfp_pdf_path: str,
    context_pdf_paths: List[str],
    work_dir: Path,
    progress_cb: Optional[Any] = None,
    requests_per_minute: int = 8,
    max_questions_this_run: Optional[int] = None,
    resume_from_partial: bool = True,
) -> Dict[str, Any]:
    wf_dir = ensure_dirs(work_dir)

    # Models
    llm: LLM = Gemini(model="gemini-2.5-flash", timeout=600, max_output_tokens=8192)
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    parser = build_parser()

    # Parse context documents
    if progress_cb:
        progress_cb({"phase": "parse_context_start", "total": len(context_pdf_paths), "current": 0})
    file_dicts: Dict[str, Dict[str, Any]] = {}
    for idx, file_name in enumerate(context_pdf_paths):
        try:
            results_any = await aparse_with_retry(parser, [file_name])
        except Exception:
            if progress_cb:
                progress_cb({"phase": "parse_context_progress", "file": file_name, "total": len(context_pdf_paths), "current": idx + 1})
            continue
        result_obj = None
        if isinstance(results_any, list) and results_any:
            result_obj = results_any[0]
        elif not isinstance(results_any, list):
            result_obj = results_any
        if result_obj is None:
            if progress_cb:
                progress_cb({"phase": "parse_context_progress", "file": file_name, "total": len(context_pdf_paths), "current": idx + 1})
            continue
        try:
            docs = result_obj.get_markdown_nodes(split_by_page=True)
        except Exception:
            docs = []
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
            file_dicts[fpath]["summary"] = summarize_file_nodes(llm, file_dicts[fpath]["docs"]) 
        except Exception:
            file_dicts[fpath]["summary"] = ""
        if progress_cb:
            progress_cb({"phase": "summarize_progress", "file": fpath, "total": len(valid_files), "current": sidx + 1})

    # Build vector store
    persist_dir = str(work_dir / "storage_chroma_rfp")
    index: VectorStoreIndex = build_index(persist_dir, embed_model)

    all_nodes: List[TextNode] = [c for d in file_dicts.values() for c in d["docs"]]
    if all_nodes:
        index.insert_nodes(all_nodes)
    if progress_cb:
        progress_cb({"phase": "index_built", "nodes": len(all_nodes)})

    summaries = {k: v.get("summary", "") for k, v in file_dicts.items()}
    tools = generate_tools(index, list(file_dicts.keys()), summaries)

    # Parse RFP template
    if progress_cb:
        progress_cb({"phase": "parse_rfp_start"})
    rfp_result = await aparse_with_retry(parser, rfp_pdf_path)
    rfp_docs = rfp_result.get_markdown_nodes(split_by_page=True)
    if progress_cb:
        progress_cb({"phase": "parse_rfp_done", "pages": len(rfp_docs)})

    # Extract questions
    all_text = "\n\n".join([d.get_content(metadata_mode="all") for d in rfp_docs])
    prompt = PromptTemplate(template=EXTRACT_KEYS_PROMPT)
    file_metadata = "\n\n".join([
        f"Name:{t.metadata.name}\nDescription:{t.metadata.description}" for t in tools
    ])
    class OutputQuestions(BaseModel):
        questions: List[str]

    try:
        questions = (
            await llm.astructured_predict(
                OutputQuestions,
                prompt,
                file_metadata=file_metadata,
                rfp_text=all_text,
            )
        ).questions
    except Exception:
        # fallback to unstructured parse
        try:
            raw = await llm.apredict(prompt, file_metadata=file_metadata, rfp_text=all_text)
            questions = [q.strip("*- ") for q in str(raw).splitlines() if q.strip()]
        except Exception:
            questions = []

    if progress_cb:
        progress_cb({"phase": "questions_extracted", "total": len(questions)})

    # Persist questions immediately
    try:
        with open(wf_dir / "all_keys.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(questions))
    except Exception:
        pass

    # Answer questions
    answers: List[Dict[str, str]] = []
    research_agent = FunctionAgent(tools=tools, llm=llm, system_prompt="You are a research agent.")
    try:
        with open(wf_dir / "combined_answers.jsonl", "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass

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

    remaining_qs: List[str] = [q for q in questions if q not in answered_set]
    if max_questions_this_run is not None and max_questions_this_run > 0:
        remaining_qs = remaining_qs[:max_questions_this_run]

    min_gap_sec = max(0.0, 60.0 / max(1, requests_per_minute))
    last_call_ts: Optional[float] = None

    for qidx, q in enumerate(remaining_qs):
        if last_call_ts is not None:
            import time
            gap = time.time() - last_call_ts
            if gap < min_gap_sec:
                await asyncio.sleep(min_gap_sec - gap)
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await research_agent.run(q)
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    import re
                    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}", msg)
                    wait_s = int(m.group(1)) if m else 35
                    if progress_cb:
                        progress_cb({"phase": "rate_limited", "wait_seconds": wait_s})
                    await asyncio.sleep(wait_s)
                    if attempt < 4:
                        continue
                qa_err = {"question": q, "answer": "", "error": msg}
                answers.append(qa_err)
                append_jsonl(wf_dir / "combined_answers.jsonl", qa_err)
                if progress_cb:
                    progress_cb({"phase": "qa_progress", "current": qidx + 1, "total": len(remaining_qs)})
                resp = None
                break
        import time
        last_call_ts = time.time()

        if resp is not None:
            qa = {"question": q, "answer": str(resp)}
            answers.append(qa)
            append_jsonl(wf_dir / "combined_answers.jsonl", qa)
        if progress_cb:
            progress_cb({"phase": "qa_progress", "current": qidx + 1, "total": len(remaining_qs)})

    # Generate output in small sections
    q_to_a: Dict[str, str] = {a.get("question", ""): a.get("answer", "") for a in answers}
    section_md_list: List[str] = []
    sec_prompt = PromptTemplate(GENERATE_SECTION_PROMPT)
    if progress_cb:
        progress_cb({"phase": "generating_sections_start", "total": len(questions)})
    for sidx, q in enumerate(questions):
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
            progress_cb({"phase": "generating_sections_progress", "current": sidx + 1, "total": len(questions)})

    final_output = "\n\n".join(section_md_list)

    with open(wf_dir / "final_output.md", "w", encoding="utf-8") as f:
        f.write(final_output)

    return {
        "questions": questions,
        "answers": answers,
        "markdown": final_output,
        "output_dir": str(wf_dir),
    }
