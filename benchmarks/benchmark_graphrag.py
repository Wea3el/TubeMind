"""Benchmark TubeMind's GraphRAG backends on a tiny transcript-like corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


CORPUS = [
    (
        "Title: Budget Headphones Review\n"
        "URL: https://www.youtube.com/watch?v=demo001\n\n"
        "The reviewer compares the SoundCore Life Q30, Sony CH520, and JBL Tune 720BT. "
        "They say the SoundCore has the best noise cancellation under $100, while Sony has the clearest calls. "
        "The JBL option has stronger bass but weaker microphone quality."
    ),
    (
        "Title: Best Cheap Wireless Headphones\n"
        "URL: https://www.youtube.com/watch?v=demo002\n\n"
        "For buyers under $100, comfort and battery life matter more than premium codecs. "
        "The video recommends SoundCore for travel, Sony for office calls, and JBL for gym listening. "
        "It warns that very cheap no-name headphones often have poor app support."
    ),
]

METADATA = [
    {
        "doc_id": "youtube:demo001",
        "video_id": "demo001",
        "title": "Budget Headphones Review",
        "url": "https://www.youtube.com/watch?v=demo001",
        "thumbnail": "",
        "channel_title": "Demo Channel",
    },
    {
        "doc_id": "youtube:demo002",
        "video_id": "demo002",
        "title": "Best Cheap Wireless Headphones",
        "url": "https://www.youtube.com/watch?v=demo002",
        "thumbnail": "",
        "channel_title": "Demo Channel",
    },
]


@dataclass(slots=True)
class BenchmarkResult:
    """Capture one measured backend run in a JSON-friendly shape.

    The benchmark measures ingestion and query time separately because GraphRAG
    libraries often make different tradeoffs between expensive indexing and
    cheap retrieval. Keeping the answer length and chunk count beside timings
    helps catch failed or empty runs that would otherwise look artificially
    fast.
    """

    backend: str
    insert_seconds: float
    query_seconds: float
    total_seconds: float
    answer_chars: int
    chunk_count: int


def _openai_model() -> str:
    """Return the chat model used by both benchmark backends.

    TubeMind centralizes its normal model choice in ``OPENAI_MODEL``. The
    benchmark mirrors that setting so a local speed comparison reflects the
    application's real LLM latency rather than a hard-coded model mismatch.
    """

    return os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")


def _embedding_model() -> str:
    """Return the embedding model used by the Fast GraphRAG benchmark.

    Fast GraphRAG requires an explicit embedding service when we customize the
    chat model. Matching the application default keeps the benchmark aligned
    with production retrieval behavior while still allowing experiments through
    environment variables.
    """

    return os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


async def _benchmark_fast_graphrag(working_dir: Path, question: str) -> BenchmarkResult:
    """Measure Fast GraphRAG ingestion and query time for the sample corpus.

    This function uses the same public API as TubeMind's runtime adapter:
    ``async_insert`` with YouTube source metadata followed by ``async_query``
    with references enabled. It is intentionally small enough to run manually
    without creating a large OpenAI bill, but it still exercises entity
    extraction, vector indexing, graph persistence, and retrieval.
    """

    from fast_graphrag import GraphRAG, QueryParam
    from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService

    rag = GraphRAG(
        working_dir=str(working_dir),
        domain=(
            "Analyze YouTube transcript evidence for a board-scoped research question. "
            "Identify videos, creators, products, claims, comparisons, and supporting evidence."
        ),
        example_queries="\n".join(
            [
                "What are the strongest recommendations across these videos?",
                "Which tradeoffs do reviewers compare?",
                "What source evidence supports the answer?",
            ]
        ),
        entity_types=["Video", "Product", "Feature", "Claim", "Comparison", "Recommendation", "Evidence"],
        config=GraphRAG.Config(
            llm_service=OpenAILLMService(model=_openai_model()),
            embedding_service=OpenAIEmbeddingService(
                model=_embedding_model(),
                embedding_dim=int(os.environ.get("OPENAI_EMBEDDING_DIM", "1536")),
            ),
        ),
    )

    start = time.perf_counter()
    await rag.async_insert(CORPUS, metadata=METADATA, show_progress=False)
    inserted = time.perf_counter()
    response = await rag.async_query(question, params=QueryParam(with_references=True))
    queried = time.perf_counter()
    chunks = list(getattr(getattr(response, "context", None), "chunks", []) or [])
    return BenchmarkResult(
        backend="fast-graphrag",
        insert_seconds=inserted - start,
        query_seconds=queried - inserted,
        total_seconds=queried - start,
        answer_chars=len(str(getattr(response, "response", "") or "")),
        chunk_count=len(chunks),
    )


async def _benchmark_lightrag(working_dir: Path, question: str) -> BenchmarkResult:
    """Measure the previous LightRAG integration for an apples-to-apples baseline.

    LightRAG is no longer a project dependency after the migration, so this
    function is optional and imports it only when the caller runs the script
    with a temporary dependency such as ``uv run --with lightrag-hku``. It uses
    the same old TubeMind insertion/query calls so the result can validate
    whether Fast GraphRAG is actually faster on the same tiny corpus.
    """

    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=partial(openai_complete_if_cache, _openai_model()),
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()

    start = time.perf_counter()
    track_id = await rag.ainsert(
        CORPUS,
        ids=[item["doc_id"] for item in METADATA],
        file_paths=[item["url"] for item in METADATA],
    )
    if hasattr(rag, "doc_status"):
        await rag.doc_status.get_docs_by_track_id(track_id)
    inserted = time.perf_counter()
    answer = await rag.aquery(question, param=QueryParam(mode="mix", response_type="Multiple Paragraphs"))
    data = await rag.aquery_data(question, param=QueryParam(mode="mix"))
    queried = time.perf_counter()
    await rag.finalize_storages()
    chunks = list((data or {}).get("data", {}).get("chunks", []) or [])
    return BenchmarkResult(
        backend="lightrag",
        insert_seconds=inserted - start,
        query_seconds=queried - inserted,
        total_seconds=queried - start,
        answer_chars=len(str(answer or "")),
        chunk_count=len(chunks),
    )


def _summarize(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Aggregate repeated benchmark runs by backend.

    LLM-backed timings are noisy because they include network and provider
    queueing latency. Reporting medians across repeats makes the output more
    useful while preserving every raw run for debugging or comparison.
    """

    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.backend, []).append(result)

    summary: dict[str, Any] = {}
    for backend, backend_results in grouped.items():
        summary[backend] = {
            "runs": [asdict(result) for result in backend_results],
            "median_insert_seconds": statistics.median(result.insert_seconds for result in backend_results),
            "median_query_seconds": statistics.median(result.query_seconds for result in backend_results),
            "median_total_seconds": statistics.median(result.total_seconds for result in backend_results),
        }

    if "fast-graphrag" in summary and "lightrag" in summary:
        light_total = summary["lightrag"]["median_total_seconds"]
        fast_total = summary["fast-graphrag"]["median_total_seconds"]
        summary["comparison"] = {
            "fast_graphrag_total_seconds": fast_total,
            "lightrag_total_seconds": light_total,
            "speedup": light_total / fast_total if fast_total else None,
            "fast_graphrag_is_faster": fast_total < light_total,
        }
    return summary


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected backend benchmarks and return structured results.

    Every repeat receives a fresh temporary working directory so persistent
    graph state cannot make later runs faster than first-use application
    behavior. The caller can choose ``fast``, ``lightrag``, or ``both`` to keep
    the normal dependency set lean while still allowing historical comparison.
    """

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the GraphRAG benchmark.")

    selected = ["fast-graphrag", "lightrag"] if args.backend == "both" else [args.backend]
    results: list[BenchmarkResult] = []
    with tempfile.TemporaryDirectory(prefix="tubemind-graphrag-bench-") as root:
        root_path = Path(root)
        for repeat in range(args.repeats):
            for backend in selected:
                working_dir = root_path / f"{backend}-{repeat}"
                working_dir.mkdir(parents=True, exist_ok=True)
                if backend == "fast-graphrag":
                    results.append(await _benchmark_fast_graphrag(working_dir, args.question))
                else:
                    results.append(await _benchmark_lightrag(working_dir, args.question))
    return _summarize(results)


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for the GraphRAG benchmark.

    The defaults are intentionally conservative: one Fast GraphRAG run over two
    short transcript snippets. Use ``--backend both`` and run through
    ``uv run --with lightrag-hku`` when a direct comparison with the retired
    backend is needed.
    """

    parser = argparse.ArgumentParser(description="Benchmark TubeMind GraphRAG backends.")
    parser.add_argument("--backend", choices=["fast-graphrag", "lightrag", "both"], default="fast-graphrag")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--question", default="What are the best headphones under $100 and why?")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark from the command line and print machine-readable JSON."""

    args = _parse_args()
    print(json.dumps(asyncio.run(_run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
