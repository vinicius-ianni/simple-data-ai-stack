# zvec Healthcare Demo

Semantic patient search using an embedded vector database. Describe a clinical
presentation, get the most similar cases ranked by meaning — not keywords.

Runs entirely offline. No server, no Docker, no cloud.

---

## What it does

1. Generates 10,000 synthetic patient records across 8 departments
2. Embeds each clinical note using `fastembed` (ONNX, no PyTorch)
3. Indexes everything in `zvec` — an in-process vector DB built on Alibaba's Proxima engine
4. Runs search demos: semantic similarity, age/severity filters, department scoping
5. Benchmarks 4 approaches side-by-side: FAISS · zvec · ChromaDB · NumPy

## How to run

```bash
./run.sh
```

Or directly:

```bash
uv run --python 3.12 zvec_healthcare_demo.py
```

Requires [uv](https://docs.astral.sh/uv/). Everything else (dependencies, Python 3.12, ONNX model) is handled automatically.

First run downloads the BGE-small-en-v1.5 model (~23 MB). After that it's fully offline.

---

## Benchmark summary

| Library | Latency | Persist | Filter |
|---|---|---|---|
| FAISS HNSW | ~0.04ms | ✗ | ✗ |
| NumPy | ~0.18ms | ✗ | ✗ |
| zvec HNSW | ~0.31ms | ✓ disk | ✓ native |
| ChromaDB | ~0.45ms | optional | ✓ |

**FAISS is faster** — it does less. No persistence, no filtering. Rebuild every run.

**NumPy beats zvec at 10K vectors** — HNSW overhead only pays off at scale. At 1M records the projection flips to ~38x faster.

**zvec vs ChromaDB** is the fair comparison (same feature tier). zvec wins clearly on filtered queries: 0.5ms vs 10ms+.

---

## Stack

| Tool | Role |
|---|---|
| [zvec](https://github.com/alibaba/proxima) | Embedded vector DB — HNSW, InvertIndex, disk persistence |
| [fastembed](https://github.com/qdrant/fastembed) | ONNX embeddings, no PyTorch |
| [polars](https://pola.rs) | Rust DataFrames for patient data |
| [faiss-cpu](https://github.com/facebookresearch/faiss) | Benchmark baseline |
| [chromadb](https://www.trychroma.com) | Benchmark baseline |
| [rich](https://github.com/Textualize/rich) | Terminal UI |
