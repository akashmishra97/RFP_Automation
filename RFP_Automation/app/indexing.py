from pathlib import Path
from typing import Dict, List, Optional
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import TextNode


def build_index(persist_dir: str, embed_model) -> VectorStoreIndex:
    vector_store = ChromaVectorStore.from_params(collection_name="rfp_docs", persist_dir=persist_dir)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


def generate_tools(index: VectorStoreIndex, files: List[str], summaries: Dict[str, str]) -> List[FunctionTool]:
    def generate_tool(file: str, file_description: Optional[str] = None) -> FunctionTool:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="file_path", operator=FilterOperator.EQ, value=file)]
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
        tools.append(generate_tool(f, summaries.get(f, "")))
    return tools


def summarize_file_nodes(llm, docs: List[TextNode]) -> str:
    idx = SummaryIndex(docs)
    resp = idx.as_query_engine(llm=llm).query(
        "Generate a short 1-2 line summary of this file to help inform an agent on what this file is about."
    )
    return str(resp)
