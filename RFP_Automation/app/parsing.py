from typing import Any, List, Optional
from llama_cloud_services import LlamaParse


def build_parser() -> LlamaParse:
    return LlamaParse(
        parse_mode="parse_page_with_agent",
        model="gemini-2.5-flash",
        high_res_ocr=True,
        outlined_table_extraction=True,
        output_tables_as_HTML=True,
    )


async def aparse_with_retry(parser: LlamaParse, inputs: Any, max_attempts: int = 4, base_delay: int = 5):
    import asyncio

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await parser.aparse(inputs)
        except Exception as e:
            last_exc = e
            await asyncio.sleep(base_delay * attempt)
    assert last_exc is not None
    raise last_exc
