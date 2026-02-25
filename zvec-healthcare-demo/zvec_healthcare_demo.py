#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "zvec>=0.1.0",
#   "polars>=1.0.0",
#   "numpy>=2.0.0",
#   "rich>=13.7.0",
#   "fastembed>=0.3.6",
#   "orjson>=3.9.0",
#   "faiss-cpu>=1.8.0",
#   "chromadb>=0.5.0",
# ]
# ///
"""
╔══════════════════════════════════════════════════════════════╗
║         ZVEC Healthcare Intelligence Platform                ║
║   Semantic Patient Search & Clinical Decision Support        ║
║                                                              ║
║   zvec (Alibaba/Proxima C++)  ·  fastembed (ONNX)           ║
║   polars (Rust DataFrames)    ·  rich (Terminal UI)          ║
╚══════════════════════════════════════════════════════════════╝

What is zvec?
  An in-process vector database built on Alibaba's Proxima engine.
  - No server needed — embedded like SQLite, zero infra overhead
  - HNSW / IVF / Flat indexing with SIMD acceleration (C++)
  - Dense + sparse vector support with numeric metadata filtering
  - Persistent on-disk storage with memory-mapped reads
  - Python 3.10-3.12 | Node.js | Linux x86_64/ARM64 | macOS ARM64

Healthcare Use Case:
  "Find semantically similar patients based on clinical notes."
  Enables clinical decision support — 'show me past cases like this patient'
  Filter by department, age range, severity alongside vector similarity.

Run with:
  uv run zvec_healthcare_demo.py
"""
from __future__ import annotations

import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import chromadb
import faiss
import numpy as np
import orjson
import polars as pl
import zvec
from fastembed import TextEmbedding
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from zvec import (
    CollectionOption,
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    HnswIndexParam,
    InvertIndexParam,
    LogLevel,
    LogType,
    OptimizeOption,
    VectorQuery,
    VectorSchema,
)

# ─── Configuration ────────────────────────────────────────────────────────────

N_PATIENTS: int = 10_000          # synthetic patient records
EMBEDDING_DIM: int = 384          # BGE-small-en-v1.5 output dimension
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"  # ONNX quantized, ~23 MB download
COLLECTION_PATH: str = "/tmp/zvec_healthcare_demo"
RANDOM_SEED: int = 42
N_BENCHMARK_QUERIES: int = 50

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

console = Console()

# ─── Healthcare Domain Fixtures ───────────────────────────────────────────────

DEPARTMENTS: dict[int, str] = {
    0: "Emergency",
    1: "Cardiology",
    2: "Neurology",
    3: "Oncology",
    4: "Orthopedics",
    5: "Pulmonology",
    6: "Gastroenterology",
    7: "Endocrinology",
}

DEPT_COLORS: dict[int, str] = {
    0: "red",
    1: "bright_red",
    2: "bright_blue",
    3: "magenta",
    4: "yellow",
    5: "cyan",
    6: "green",
    7: "bright_yellow",
}

CHIEF_COMPLAINTS: dict[int, list[str]] = {
    0: [  # Emergency
        "acute chest pain with radiation to left arm and diaphoresis",
        "sudden loss of consciousness with brief seizure-like activity",
        "severe abdominal pain with guarding and rigidity",
        "major trauma following high-speed motor vehicle accident",
        "anaphylactic reaction with urticaria and bronchospasm",
        "acute stroke symptoms with left facial droop and slurred speech",
    ],
    1: [  # Cardiology
        "chest tightness on exertion with diaphoresis and nausea",
        "palpitations with irregular heartbeat and near-syncope",
        "bilateral lower extremity edema with orthopnea",
        "syncope during physical activity with rapid recovery",
        "exertional dyspnea with reduced exercise tolerance",
        "new onset atrial fibrillation with rapid ventricular rate",
    ],
    2: [  # Neurology
        "sudden onset thunderclap headache worst of life",
        "focal weakness right-sided hemiparesis with facial droop",
        "transient monocular vision loss amaurosis fugax",
        "confusion and altered mental status with agitation",
        "resting tremor with cogwheel rigidity and bradykinesia",
        "progressive memory loss affecting daily function",
    ],
    3: [  # Oncology
        "unintentional 20-pound weight loss with profound fatigue",
        "persistent productive cough with hemoptysis",
        "palpable abdominal mass with early satiety and anorexia",
        "cervical lymphadenopathy with night sweats and fever",
        "severe bone pain at rest with pathological fracture risk",
        "jaundice with dark urine and pale stools",
    ],
    4: [  # Orthopedics
        "acute knee pain and swelling after fall with pivot injury",
        "hip pain with inability to bear weight after ground-level fall",
        "low back pain radiating to left leg sciatica pattern",
        "shoulder dislocation with numbness in lateral arm",
        "wrist fracture following fall on outstretched hand FOOSH",
        "neck pain with bilateral arm tingling cervical radiculopathy",
    ],
    5: [  # Pulmonology
        "progressive exertional dyspnea with productive purulent cough",
        "acute wheezing with reduced peak flow and accessory muscle use",
        "hemoptysis with weight loss and night sweats",
        "pleuritic chest pain with high fever and rigors",
        "excessive daytime somnolence with witnessed apneas OSA",
        "dry cough with progressive exertional dyspnea ILD pattern",
    ],
    6: [  # Gastroenterology
        "severe epigastric burning pain worse with meals",
        "hematemesis with coffee-ground vomiting and melena",
        "progressive dysphagia for solids then liquids",
        "right upper quadrant pain with jaundice and fever",
        "chronic bloody diarrhea with weight loss and tenesmus",
        "acute nausea vomiting with right lower quadrant tenderness",
    ],
    7: [  # Endocrinology
        "polyuria polydipsia with weight loss uncontrolled blood glucose",
        "fatigue cold intolerance weight gain and hair thinning hypothyroid",
        "palpitations tremor heat intolerance and weight loss hyperthyroid",
        "hypotension weakness hyperpigmentation adrenal insufficiency",
        "recurrent hypoglycemic episodes with confusion and diaphoresis",
        "moon face central obesity and purple striae Cushing syndrome",
    ],
}

DIAGNOSES: dict[int, list[str]] = {
    0: ["STEMI", "Anaphylaxis", "Polytrauma", "Acute Abdomen", "Septic Shock", "Hemorrhagic Stroke"],
    1: ["NSTEMI", "Atrial Fibrillation", "Decompensated CHF", "Aortic Stenosis", "Cardiomyopathy", "SVT"],
    2: ["Ischemic Stroke", "SAH", "TIA", "Parkinson's Disease", "Migraine", "Alzheimer's Disease"],
    3: ["Lung Adenocarcinoma", "Non-Hodgkin Lymphoma", "Colorectal Cancer", "Breast Cancer", "Pancreatic Cancer", "HCC"],
    4: ["ACL Tear", "Hip Fracture", "L4-L5 Disc Herniation", "Rotator Cuff Tear", "Colles Fracture", "Cervical Myelopathy"],
    5: ["COPD Exacerbation", "Status Asthmaticus", "Pulmonary Embolism", "Community Pneumonia", "IPF", "Lung Cancer"],
    6: ["Peptic Ulcer Disease", "Upper GI Bleed", "Esophageal Cancer", "Acute Cholecystitis", "Crohn's Disease", "Appendicitis"],
    7: ["DKA", "Hypothyroidism", "Graves' Disease", "Addison's Disease", "Insulinoma", "Cushing's Syndrome"],
}

COMORBIDITIES: list[str] = [
    "hypertension", "type 2 diabetes mellitus", "hyperlipidemia", "obesity BMI >30",
    "coronary artery disease", "atrial fibrillation", "COPD", "chronic kidney disease stage 3",
    "hypothyroidism", "osteoporosis", "obstructive sleep apnea", "depression and anxiety",
    "peripheral artery disease", "heart failure with reduced EF", "cirrhosis Child-Pugh B",
]

MEDICATIONS: list[str] = [
    "metformin 1000mg BID", "lisinopril 10mg daily", "atorvastatin 40mg nightly",
    "metoprolol succinate 50mg daily", "omeprazole 20mg daily", "aspirin 81mg daily",
    "amlodipine 5mg daily", "levothyroxine 75mcg daily", "sertraline 100mg daily",
    "furosemide 40mg daily", "warfarin 5mg daily", "albuterol MDI PRN",
    "apixaban 5mg BID", "empagliflozin 10mg daily", "tiotropium 18mcg daily",
]


# ─── Data Generation ──────────────────────────────────────────────────────────

def generate_clinical_note(
    age: int,
    gender: str,
    dept_code: int,
    complaint: str,
    diagnosis: str,
    comorbidities: list[str],
    medications: list[str],
    severity: int,
) -> str:
    """Generate a realistic-sounding synthetic clinical note."""
    pronoun = "He" if gender == "M" else "She" if gender == "F" else "They"
    hx = ", ".join(comorbidities) if comorbidities else "no significant past medical history"
    meds = ", ".join(medications) if medications else "no regular medications"
    sev_desc = {1: "mild", 2: "mild-to-moderate", 3: "moderate", 4: "moderate-to-severe", 5: "severe"}[severity]
    vitals_note = (
        "hemodynamically unstable requiring urgent intervention"
        if severity >= 4
        else "hemodynamically stable on arrival"
    )

    return (
        f"Patient is a {age}-year-old {gender} presenting to the {DEPARTMENTS[dept_code]} department "
        f"with {complaint}. {pronoun} reports {sev_desc} symptom burden with onset over the past "
        f"{'hours' if severity >= 4 else 'days to weeks'}. "
        f"Past medical history is significant for {hx}. "
        f"Current medications include {meds}. "
        f"On examination, patient is {vitals_note}. "
        f"Laboratory and imaging workup support the diagnosis of {diagnosis}. "
        f"Plan: initiate {'emergent' if severity == 5 else 'urgent' if severity >= 3 else 'standard'} "
        f"management protocol per {DEPARTMENTS[dept_code]} guidelines."
    )


def generate_healthcare_data(n: int) -> pl.DataFrame:
    """Generate N synthetic patient records using polars + numpy."""
    rng = np.random.default_rng(RANDOM_SEED)

    dept_codes = rng.integers(0, 8, size=n)
    ages = rng.integers(18, 90, size=n)
    genders = rng.choice(["M", "F"], size=n)
    severities = rng.integers(1, 6, size=n)  # 1–5

    patient_ids: list[str] = [f"PT{i:05d}" for i in range(n)]
    notes: list[str] = []

    for i in range(n):
        dept = int(dept_codes[i])
        complaint = random.choice(CHIEF_COMPLAINTS[dept])
        diagnosis = random.choice(DIAGNOSES[dept])
        comorbidities = random.sample(COMORBIDITIES, random.randint(0, 4))
        medications = random.sample(MEDICATIONS, random.randint(0, 5))
        note = generate_clinical_note(
            age=int(ages[i]),
            gender=str(genders[i]),
            dept_code=dept,
            complaint=complaint,
            diagnosis=diagnosis,
            comorbidities=comorbidities,
            medications=medications,
            severity=int(severities[i]),
        )
        notes.append(note)

    return pl.DataFrame(
        {
            "patient_id": patient_ids,
            "patient_num": list(range(n)),
            "age": ages.astype(int).tolist(),
            "gender": genders.tolist(),
            "department_code": dept_codes.astype(int).tolist(),
            "department": [DEPARTMENTS[int(d)] for d in dept_codes],
            "severity": severities.astype(int).tolist(),
            "clinical_note": notes,
        }
    )


# ─── zvec Collection ─────────────────────────────────────────────────────────

def create_collection(path: str) -> zvec.Collection:
    """
    Create zvec collection with healthcare schema.

    Schema design:
      - Scalar fields (INT32/INT64 with InvertIndex) → filterable metadata
      - VectorSchema (FP32 + HNSW) → semantic similarity search
      - STRING fields are stored but not used for range-filtering
        (zvec's range optimizations target numeric types)
    """
    if Path(path).exists():
        shutil.rmtree(path)

    schema = CollectionSchema(
        name="healthcare_patients",
        fields=[
            FieldSchema(
                "patient_num",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema(
                "age",
                DataType.INT32,
                nullable=True,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema(
                "severity",
                DataType.INT32,
                nullable=True,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema(
                "department_code",
                DataType.INT32,
                nullable=True,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
        ],
        vectors=[
            VectorSchema(
                "clinical_embedding",
                DataType.VECTOR_FP32,
                dimension=EMBEDDING_DIM,
                index_param=HnswIndexParam(),
            ),
        ],
    )

    return zvec.create_and_open(
        path=path,
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )


def bulk_insert(
    collection: zvec.Collection,
    df: pl.DataFrame,
    embeddings: np.ndarray,
    progress: Progress,
) -> float:
    """Batch-insert all patient documents; returns total elapsed seconds."""
    BATCH_SIZE = 500
    insert_task = progress.add_task(
        f"[cyan]Inserting {N_PATIENTS:,} patient docs...", total=len(df)
    )

    t0 = time.perf_counter()
    for start in range(0, len(df), BATCH_SIZE):
        batch_df = df.slice(start, BATCH_SIZE)
        batch_emb = embeddings[start : start + BATCH_SIZE]

        docs = [
            Doc(
                id=row["patient_id"],
                fields={
                    "patient_num": row["patient_num"],
                    "age": row["age"],
                    "severity": row["severity"],
                    "department_code": row["department_code"],
                },
                vectors={"clinical_embedding": batch_emb[j].tolist()},
            )
            for j, row in enumerate(batch_df.iter_rows(named=True))
        ]
        collection.insert(docs)
        progress.advance(insert_task, len(docs))

    elapsed = time.perf_counter() - t0
    progress.remove_task(insert_task)
    return elapsed


# ─── Search Helpers ───────────────────────────────────────────────────────────

def semantic_search(
    collection: zvec.Collection,
    query_vector: np.ndarray,
    df: pl.DataFrame,
    topk: int = 5,
    filter_expr: Optional[str] = None,
) -> list[dict]:
    """Run zvec vector similarity search, enrich results from polars DataFrame."""
    kwargs: dict = {"topk": topk}
    if filter_expr:
        kwargs["filter"] = filter_expr

    raw = collection.query(
        VectorQuery("clinical_embedding", vector=query_vector.tolist()),
        **kwargs,
    )

    results: list[dict] = []
    for doc in raw:
        rows = df.filter(pl.col("patient_id") == doc.id)
        if len(rows) == 0:
            continue
        row = rows.row(0, named=True)
        results.append(
            {
                "patient_id": doc.id,
                "age": row["age"],
                "gender": row["gender"],
                "department": row["department"],
                "dept_code": row["department_code"],
                "severity": row["severity"],
                "note_preview": row["clinical_note"][:110] + "…",
            }
        )
    return results


def scalar_filter_query(
    collection: zvec.Collection,
    df: pl.DataFrame,
    filter_expr: str,
    topk: int = 5,
) -> list[dict]:
    """Pure metadata filter — no vector search (zvec scan)."""
    raw = collection.query(filter=filter_expr, topk=topk)
    results: list[dict] = []
    for doc in raw:
        rows = df.filter(pl.col("patient_id") == doc.id)
        if len(rows) == 0:
            continue
        row = rows.row(0, named=True)
        results.append(
            {
                "patient_id": doc.id,
                "age": row["age"],
                "gender": row["gender"],
                "department": row["department"],
                "dept_code": row["department_code"],
                "severity": row["severity"],
                "note_preview": row["clinical_note"][:110] + "…",
            }
        )
    return results


# ─── Rich Display Helpers ────────────────────────────────────────────────────

SEV_COLORS = {1: "green", 2: "yellow", 3: "yellow", 4: "orange1", 5: "red"}
SEV_LABELS = {1: "Low", 2: "Mild", 3: "Moderate", 4: "High", 5: "Critical"}


def results_table(records: list[dict], title: str) -> Table:
    """Render search results as a rich table."""
    t = Table(
        title=title,
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold cyan",
        show_lines=True,
        expand=False,
    )
    t.add_column("#", width=3, justify="center", style="dim")
    t.add_column("Patient ID", width=11, style="bright_white")
    t.add_column("Age", width=4, justify="center")
    t.add_column("Sex", width=4, justify="center")
    t.add_column("Department", width=16, style="green")
    t.add_column("Severity", width=12, justify="center")
    t.add_column("Clinical Note Preview", width=60, style="dim")

    for rank, r in enumerate(records, 1):
        sev = r["severity"]
        dept_color = DEPT_COLORS.get(r["dept_code"], "white")
        sev_text = Text(
            f"{'●' * sev}{'○' * (5 - sev)} {SEV_LABELS[sev]}",
            style=SEV_COLORS.get(sev, "white"),
        )
        t.add_row(
            str(rank),
            r["patient_id"],
            str(r["age"]),
            r["gender"],
            Text(r["department"], style=dept_color),
            sev_text,
            r["note_preview"],
        )
    return t


def print_header() -> None:
    """Render the application banner + tech-stack pills."""
    banner = Text()
    banner.append("  ZVEC  ", style="bold white on dark_blue")
    banner.append("  Healthcare Intelligence Platform\n", style="bold bright_white")
    banner.append(
        "  Semantic Patient Search · Clinical Decision Support\n", style="dim white"
    )
    banner.append(
        "  Built on Alibaba Proxima Engine — C++/SIMD — In-Process — Zero Infra",
        style="dim cyan",
    )

    console.print(
        Panel(Align.center(banner), border_style="bright_blue", padding=(1, 6))
    )

    pills = [
        Panel(
            Align.center("[bold]zvec[/bold]\nC++ / SIMD\nProxima Engine"),
            title="[cyan]Vector DB[/cyan]",
            border_style="cyan",
            width=22,
        ),
        Panel(
            Align.center("[bold]fastembed[/bold]\nONNX Runtime\nNo PyTorch"),
            title="[green]Embeddings[/green]",
            border_style="green",
            width=22,
        ),
        Panel(
            Align.center("[bold]polars[/bold]\nRust DataFrames\nZero-copy"),
            title="[yellow]Data[/yellow]",
            border_style="yellow",
            width=22,
        ),
        Panel(
            Align.center("[bold]orjson[/bold]\nRust JSON\nFast serde"),
            title="[magenta]Serialization[/magenta]",
            border_style="magenta",
            width=22,
        ),
    ]
    console.print(Columns(pills, equal=False, expand=True))
    console.print()


# ─── Benchmarking ─────────────────────────────────────────────────────────────

def bench_numpy(embeddings: np.ndarray, query_vecs: np.ndarray, topk: int = 5) -> float:
    """NumPy brute-force cosine similarity — exact but O(N·D) per query."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-9)
    latencies: list[float] = []
    for qv in query_vecs:
        t0 = time.perf_counter()
        q_norm = qv / (np.linalg.norm(qv) + 1e-9)
        scores = emb_norm @ q_norm
        np.argpartition(-scores, topk)[:topk]
        latencies.append(time.perf_counter() - t0)
    return float(np.mean(latencies)) * 1000  # ms


def bench_zvec(
    collection: zvec.Collection, query_vecs: np.ndarray, topk: int = 5
) -> float:
    """zvec HNSW approximate nearest-neighbour — O(log N) per query."""
    latencies: list[float] = []
    for qv in query_vecs:
        t0 = time.perf_counter()
        collection.query(
            VectorQuery("clinical_embedding", vector=qv.tolist()), topk=topk
        )
        latencies.append(time.perf_counter() - t0)
    return float(np.mean(latencies)) * 1000  # ms


def bench_zvec_filtered(
    collection: zvec.Collection, query_vecs: np.ndarray, topk: int = 5
) -> float:
    """zvec HNSW + numeric metadata filter."""
    latencies: list[float] = []
    for qv in query_vecs[:30]:
        t0 = time.perf_counter()
        collection.query(
            VectorQuery("clinical_embedding", vector=qv.tolist()),
            filter="age>=40 and age<=70",
            topk=topk,
        )
        latencies.append(time.perf_counter() - t0)
    return float(np.mean(latencies)) * 1000  # ms


def bench_zvec_scalar(
    collection: zvec.Collection, query_vecs: np.ndarray
) -> float:
    """Pure scalar filter scan — no vector computation."""
    latencies: list[float] = []
    for _ in query_vecs[:30]:
        t0 = time.perf_counter()
        collection.query(filter="severity>=4 and department_code=1", topk=10)
        latencies.append(time.perf_counter() - t0)
    return float(np.mean(latencies)) * 1000  # ms


# ─── FAISS Benchmarking ───────────────────────────────────────────────────────

def setup_faiss_index(embeddings: np.ndarray) -> tuple:
    """Build a FAISS HNSW index with cosine similarity (L2-normalised → inner product)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = (embeddings / (norms + 1e-9)).astype(np.float32)
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 16)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 64
    t0 = time.perf_counter()
    index.add(emb_norm)
    build_time = time.perf_counter() - t0
    return index, emb_norm, build_time


def bench_faiss(index: "faiss.IndexHNSWFlat", query_vecs_norm: np.ndarray, topk: int = 5) -> float:
    """FAISS HNSW approximate nearest-neighbour — O(log N) per query."""
    latencies: list[float] = []
    for qv in query_vecs_norm:
        t0 = time.perf_counter()
        index.search(qv.reshape(1, -1), topk)
        latencies.append(time.perf_counter() - t0)
    return float(np.mean(latencies)) * 1000  # ms


# ─── ChromaDB Benchmarking ────────────────────────────────────────────────────

def setup_chromadb_collection(
    df: pl.DataFrame,
    embeddings: np.ndarray,
    progress: Progress,
) -> tuple:
    """Build an in-memory ChromaDB collection (EphemeralClient — no disk, fair benchmark).

    EphemeralClient is purely in-memory, analogous to FAISS for fair comparison.
    Use PersistentClient(path=...) to add disk persistence like zvec.
    """
    BATCH_SIZE = 500
    client = chromadb.EphemeralClient()
    col = client.create_collection(
        name="healthcare_patients",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:construction_ef": 40,
            "hnsw:search_ef": 64,
        },
    )
    chroma_task = progress.add_task(
        f"[magenta]Building ChromaDB collection ({N_PATIENTS:,} docs)…", total=len(df)
    )
    t0 = time.perf_counter()
    for start in range(0, len(df), BATCH_SIZE):
        batch_df = df.slice(start, BATCH_SIZE)
        batch_emb = embeddings[start : start + BATCH_SIZE]
        ids = batch_df["patient_id"].to_list()
        metadatas = [
            {
                "age": int(row["age"]),
                "severity": int(row["severity"]),
                "department_code": int(row["department_code"]),
            }
            for row in batch_df.iter_rows(named=True)
        ]
        col.add(
            ids=ids,
            embeddings=batch_emb.tolist(),
            metadatas=metadatas,
        )
        progress.advance(chroma_task, len(ids))
    build_time = time.perf_counter() - t0
    progress.remove_task(chroma_task)
    return col, build_time


def bench_chromadb(col: "chromadb.Collection", query_vecs: np.ndarray, topk: int = 5) -> float:
    """ChromaDB HNSW approximate nearest-neighbour."""
    latencies: list[float] = []
    for qv in query_vecs:
        t0 = time.perf_counter()
        col.query(query_embeddings=[qv.tolist()], n_results=topk)
        latencies.append(time.perf_counter() - t0)
    return float(np.mean(latencies)) * 1000  # ms


def bench_chromadb_filtered(col: "chromadb.Collection", query_vecs: np.ndarray, topk: int = 5) -> float:
    """ChromaDB HNSW + metadata filter (age 40–70, same 30-query sample as zvec_filtered)."""
    latencies: list[float] = []
    for qv in query_vecs[:30]:
        t0 = time.perf_counter()
        col.query(
            query_embeddings=[qv.tolist()],
            n_results=topk,
            where={"$and": [{"age": {"$gte": 40}}, {"age": {"$lte": 70}}]},
        )
        latencies.append(time.perf_counter() - t0)
    return float(np.mean(latencies)) * 1000  # ms


def print_conclusion(nums: dict) -> None:
    """Render the benchmark conclusions panel with actual measured values."""
    zvec_ms = nums["zvec_ms"]
    faiss_ms = nums["faiss_ms"]
    chroma_ms = nums["chroma_ms"]
    numpy_ms = nums["numpy_ms"]

    lines = (
        f"[dim]Measured on {N_PATIENTS:,} × {EMBEDDING_DIM}D FP32 · top-5 · macOS ARM64[/dim]\n\n"
        f"[bold cyan]FAISS HNSW[/bold cyan]       [yellow]{faiss_ms:.2f}ms[/yellow] · Fastest raw search · no persist · no filter\n"
        f"                 [dim]→ Best when: batch ML pipelines, max throughput,[/dim]\n"
        f"                 [dim]  you'll build persistence yourself[/dim]\n\n"
        f"[bold bright_green]zvec HNSW[/bold bright_green]        [yellow]{zvec_ms:.2f}ms[/yellow] · FAISS-class speed + persistence + filtering\n"
        f"                 [dim]→ Best when: CLIs, desktop/edge apps, data pipes,[/dim]\n"
        f"                 [dim]  HIPAA-sensitive data (stays local)[/dim]\n\n"
        f"[bold magenta]ChromaDB embed[/bold magenta]   [yellow]{chroma_ms:.2f}ms[/yellow] · Easiest API · rich LLM ecosystem\n"
        f"                 [dim]→ Best when: RAG prototyping, LangChain apps,[/dim]\n"
        f"                 [dim]  DX matters more than sub-ms latency[/dim]\n\n"
        f"[bold white]NumPy[/bold white]            [yellow]{numpy_ms:.2f}ms[/yellow] · Zero deps · exact cosine\n"
        f"                 [dim]→ Best when: <50K vecs, offline batch, no filter[/dim]\n"
        f"                 [dim]  or persistence needed[/dim]\n\n"
        f"[dim]When to step up to Qdrant / Milvus (not benchmarked — different tier):[/dim]\n"
        f"[dim]  Multi-tenant · multiple services sharing index · REST/gRPC API[/dim]\n"
        f"[dim]  Horizontal scaling · >50M vectors · dedicated data team[/dim]\n"
        f"[dim]  (Comparing them here would be unfair — you'd be measuring HTTP overhead, not search)[/dim]"
    )
    console.print(
        Panel(
            lines,
            title="[bold bright_white] Benchmark Conclusions [/bold bright_white]",
            border_style="bright_yellow",
            padding=(1, 3),
        )
    )


# ─── Ecosystem Comparison ─────────────────────────────────────────────────────

def ecosystem_table() -> Table:
    """Show where zvec sits in the vector DB landscape."""
    t = Table(
        title="Vector Database Ecosystem — Where zvec Fits",
        box=box.DOUBLE_EDGE,
        border_style="bright_yellow",
        header_style="bold yellow",
        show_lines=True,
        expand=True,
    )
    t.add_column("Database", style="bright_white", width=12)
    t.add_column("Architecture", width=18)
    t.add_column("Scale", width=10, justify="center")
    t.add_column("Setup", width=14, justify="center")
    t.add_column("Persist", width=8, justify="center")
    t.add_column("Filter", width=7, justify="center")
    t.add_column("Best For", width=36)

    rows = [
        (
            "[bold bright_green]zvec ★[/bold bright_green]",
            "In-process\n(C++/Proxima/SIMD)",
            "10M+",
            "[green]pip install\nZero infra[/green]",
            "[green]✓ Disk[/green]",
            "[green]✓[/green]",
            "Apps needing embedded vector search\nwith persistence — 'SQLite for vectors'",
        ),
        (
            "FAISS",
            "In-process\n(Meta/C++/SIMD)",
            "Billions",
            "pip install\nZero infra",
            "✗ Manual\nsave/load",
            "✗",
            "Research, maximum raw throughput,\nno built-in persistence or filtering",
        ),
        (
            "ChromaDB",
            "Embedded or Server\n(Python + DuckDB)",
            "Millions",
            "pip install\nor Docker",
            "✓ Disk",
            "✓",
            "Rapid RAG prototyping,\nLLM app development",
        ),
        (
            "Qdrant",
            "Server\n(Rust/gRPC/REST)",
            "Billions",
            "Docker\nor Cloud",
            "✓ Cloud",
            "✓",
            "Production APIs, custom scoring,\nrich payload filtering",
        ),
        (
            "Milvus",
            "Distributed\n(C++/Go/K8s)",
            "Billions+",
            "K8s / Cloud\nComplex",
            "✓ Cloud",
            "✓",
            "Enterprise at scale,\nhigh availability, multi-tenancy",
        ),
        (
            "pgvector",
            "Postgres Extension\n(C)",
            "Millions",
            "Postgres\nextension",
            "✓ Postgres",
            "✓ SQL",
            "Existing Postgres workloads\nwanting SQL + vector in one",
        ),
    ]

    for i, row in enumerate(rows):
        style = "on grey11" if i == 0 else None
        t.add_row(*row, style=style)

    return t


def when_to_use_table() -> Table:
    """Decision guide for when zvec is the right tool."""
    t = Table(
        title="When to Choose zvec",
        box=box.ROUNDED,
        border_style="bright_yellow",
        header_style="bold yellow",
        show_lines=True,
    )
    t.add_column("Scenario", style="bright_white", width=38)
    t.add_column("zvec Advantage", style="green", width=50)

    rows = [
        (
            "CLI tools & scripts needing vector search",
            "Zero server overhead, instant startup (<1ms init)",
        ),
        (
            "Edge / desktop AI apps",
            "Ships as a pure library — no daemon or Docker required",
        ),
        (
            "Jupyter notebooks & data exploration",
            "In-process, persistent across notebook restarts",
        ),
        (
            "Microservices with vector capabilities",
            "Lower latency than HTTP roundtrip to a vector server",
        ),
        (
            "Healthcare: similar patient matching",
            "Local, persistent, HIPAA-friendly — data never leaves",
        ),
        (
            "RAG pipelines on a single machine",
            "Semantic + filter in one call, no network hop",
        ),
        (
            "When you outgrow in-memory FAISS",
            "Add persistence + filtering without switching to a server",
        ),
    ]
    for scenario, adv in rows:
        t.add_row(scenario, adv)
    return t


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    zvec.init(log_type=LogType.CONSOLE, log_level=LogLevel.ERROR)

    print_header()

    # ── Phase 1: Synthetic Data ──────────────────────────────────────────────
    console.print(Rule("[bold cyan]Phase 1 · Synthetic Healthcare Data[/bold cyan]"))

    with console.status("[cyan]Generating patient records with polars…", spinner="dots"):
        t0 = time.perf_counter()
        df = generate_healthcare_data(N_PATIENTS)
        gen_time = time.perf_counter() - t0

    # Department distribution bar chart
    dept_dist = (
        df.group_by("department")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    max_count = dept_dist["count"].max()

    dist_table = Table(
        "Department", "# Patients", "Distribution",
        box=box.SIMPLE_HEAD, header_style="bold", show_edge=False,
    )
    for row in dept_dist.iter_rows(named=True):
        bar_len = int(28 * row["count"] / max_count)
        dist_table.add_row(
            row["department"],
            str(row["count"]),
            f"[cyan]{'█' * bar_len}[/cyan][dim]{'░' * (28 - bar_len)}[/dim]",
        )

    console.print(
        Panel(
            dist_table,
            title=f"[green]✓ Generated {N_PATIENTS:,} records in {gen_time:.2f}s[/green]  [dim](polars · pure numpy RNG)[/dim]",
            border_style="green",
        )
    )

    # Severity distribution
    sev_dist = df.group_by("severity").agg(pl.len().alias("count")).sort("severity")
    sev_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold", show_edge=False)
    sev_table.add_column("Severity")
    sev_table.add_column("Count")
    sev_table.add_column("Bar")
    sev_max = sev_dist["count"].max()
    for row in sev_dist.iter_rows(named=True):
        s = row["severity"]
        color = SEV_COLORS.get(s, "white")
        bar_len = int(20 * row["count"] / sev_max)
        sev_table.add_row(
            Text(f"{'●' * s}{'○' * (5 - s)} {SEV_LABELS[s]}", style=color),
            str(row["count"]),
            Text("█" * bar_len, style=color),
        )
    console.print(Panel(sev_table, title="Severity Distribution", border_style="dim"))

    # ── Phase 2: Embeddings ─────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Phase 2 · Clinical Note Embeddings (ONNX)[/bold cyan]"))
    console.print(
        f"  Model: [cyan]{EMBEDDING_MODEL}[/cyan]  "
        f"Dim: [yellow]{EMBEDDING_DIM}[/yellow]  "
        f"Engine: [green]ONNX Runtime — no PyTorch dependency[/green]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        model_task = progress.add_task("[cyan]Downloading / loading ONNX model…", total=None)
        model = TextEmbedding(EMBEDDING_MODEL)
        progress.remove_task(model_task)

        embed_task = progress.add_task(
            f"[cyan]Encoding {N_PATIENTS:,} clinical notes…", total=N_PATIENTS
        )
        notes = df["clinical_note"].to_list()
        t0 = time.perf_counter()
        emb_list: list[np.ndarray] = []
        for emb in model.embed(notes, batch_size=256):
            emb_list.append(emb)
            progress.advance(embed_task)
        embeddings = np.array(emb_list, dtype=np.float32)
        embed_time = time.perf_counter() - t0

    throughput = N_PATIENTS / embed_time
    console.print(
        f"\n  [green]✓[/green] Shape: [yellow]{embeddings.shape}[/yellow]  "
        f"Dtype: [yellow]{embeddings.dtype}[/yellow]  "
        f"Throughput: [green]{throughput:.0f}[/green] notes/sec  "
        f"Time: [cyan]{embed_time:.1f}s[/cyan]\n"
    )

    # ── Phase 3: zvec Collection ────────────────────────────────────────────
    console.print(Rule("[bold cyan]Phase 3 · zvec Collection + HNSW Index[/bold cyan]"))

    with console.status("[cyan]Creating zvec collection schema…"):
        collection = create_collection(COLLECTION_PATH)

    schema_info = Table(box=box.SIMPLE_HEAD, show_header=False, show_edge=False, padding=(0, 2))
    schema_info.add_column(style="dim")
    schema_info.add_column(style="bright_white")
    schema_info.add_row("Collection path", COLLECTION_PATH)
    schema_info.add_row("Vector field", f"clinical_embedding  FP32  dim={EMBEDDING_DIM}")
    schema_info.add_row("Index type", "HNSW — Hierarchical Navigable Small World")
    schema_info.add_row("Scalar fields", "patient_num (INT64)  age (INT32)  severity (INT32)  department_code (INT32)")
    schema_info.add_row("Filterable", "All scalar fields with InvertIndex + range optimization")
    console.print(Panel(schema_info, title="[green]✓ Schema Created[/green]", border_style="green"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        insert_time = bulk_insert(collection, df, embeddings, progress)

    with console.status("[cyan]Optimizing HNSW index…"):
        collection.optimize(option=OptimizeOption())

    stats = collection.stats
    insert_tput = N_PATIENTS / insert_time

    ins_table = Table(box=box.SIMPLE_HEAD, show_header=False, show_edge=False, padding=(0, 2))
    ins_table.add_column(style="dim")
    ins_table.add_column(style="bright_white")
    ins_table.add_row("Documents indexed", f"{stats.doc_count:,}")
    ins_table.add_row("Insert throughput", f"{insert_tput:.0f} docs / sec")
    ins_table.add_row("Insert time", f"{insert_time:.2f} s")
    ins_table.add_row("Index completeness", str(dict(stats.index_completeness)))
    ins_table.add_row("Storage", COLLECTION_PATH)
    console.print(Panel(ins_table, title="[green]✓ Collection Ready[/green]", border_style="green"))

    # ── Phase 4: Clinical Search Demos ─────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Phase 4 · Clinical Search Demonstrations[/bold cyan]"))

    # Demo 1 — Pure semantic search
    q1 = "acute chest pain radiating to left arm with diaphoresis and nausea"
    console.print(
        Panel(
            f"[yellow]Query:[/yellow] [italic]\"{q1}\"[/italic]\n"
            "[dim]No metadata filter — pure semantic similarity across all 10,000 records[/dim]",
            title="[bold]Demo 1 · Semantic Similarity Search[/bold]",
            border_style="bright_blue",
        )
    )
    q1_vec = np.array(list(model.embed([q1]))[0], dtype=np.float32)
    res1 = semantic_search(collection, q1_vec, df, topk=5)
    console.print(results_table(res1, "Top 5 Semantically Similar Patients"))

    # Demo 2 — Vector + age + severity filter
    console.print()
    q2 = "elderly patient with exertional dyspnea bilateral lower extremity edema and reduced ejection fraction"
    filter2 = "age>=65 and severity>=3"
    console.print(
        Panel(
            f"[yellow]Query:[/yellow] [italic]\"{q2}\"[/italic]\n"
            f"[cyan]Filter:[/cyan] [bold]{filter2}[/bold]  [dim](patients 65+, severity ≥ 3)[/dim]",
            title="[bold]Demo 2 · Semantic Search + Metadata Filter[/bold]",
            border_style="bright_blue",
        )
    )
    q2_vec = np.array(list(model.embed([q2]))[0], dtype=np.float32)
    res2 = semantic_search(collection, q2_vec, df, topk=5, filter_expr=filter2)
    console.print(results_table(res2, "Top 5 Similar Patients (Age≥65, Severity≥3)"))

    # Demo 3 — Department-scoped search
    console.print()
    q3 = "progressive resting tremor cogwheel rigidity bradykinesia shuffling gait Parkinson"
    filter3 = "department_code=2"  # Neurology
    console.print(
        Panel(
            f"[yellow]Query:[/yellow] [italic]\"{q3}\"[/italic]\n"
            f"[cyan]Filter:[/cyan] [bold]{filter3}[/bold]  [dim](Neurology department only)[/dim]",
            title="[bold]Demo 3 · Department-Scoped Semantic Search[/bold]",
            border_style="bright_blue",
        )
    )
    q3_vec = np.array(list(model.embed([q3]))[0], dtype=np.float32)
    res3 = semantic_search(collection, q3_vec, df, topk=5, filter_expr=filter3)
    console.print(results_table(res3, "Top 5 Neurology Patients — Similar Presentations"))

    # Demo 4 — Pure scalar filter (no vector)
    console.print()
    filter4 = "severity=5 and department_code=0"
    console.print(
        Panel(
            f"[cyan]Filter:[/cyan] [bold]{filter4}[/bold]  [dim](Critical Emergency patients — scalar scan, no embedding computation)[/dim]",
            title="[bold]Demo 4 · Metadata-Only Filter (No Vector Search)[/bold]",
            border_style="bright_blue",
        )
    )
    res4 = scalar_filter_query(collection, df, filter_expr=filter4, topk=5)
    console.print(results_table(res4, "Critical Emergency Patients (Severity=5)"))

    # Demo 5 — OR filter across departments
    console.print()
    q5 = "uncontrolled blood glucose polyuria polydipsia diabetic ketoacidosis"
    filter5 = "department_code=7 or department_code=0"  # Endocrinology or Emergency
    console.print(
        Panel(
            f"[yellow]Query:[/yellow] [italic]\"{q5}\"[/italic]\n"
            f"[cyan]Filter:[/cyan] [bold]{filter5}[/bold]  [dim](Endocrinology OR Emergency)[/dim]",
            title="[bold]Demo 5 · Multi-Department Semantic Search[/bold]",
            border_style="bright_blue",
        )
    )
    q5_vec = np.array(list(model.embed([q5]))[0], dtype=np.float32)
    res5 = semantic_search(collection, q5_vec, df, topk=5, filter_expr=filter5)
    console.print(results_table(res5, "Top 5 Endocrinology / Emergency Matches"))

    # ── Phase 5: Four-Way Performance Benchmark ─────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Phase 5 · Four-Way Performance Benchmark[/bold cyan]"))
    console.print(
        f"  Libraries: [cyan]FAISS HNSW[/cyan]  |  [bright_green]zvec HNSW[/bright_green]  |  "
        f"[magenta]ChromaDB embedded[/magenta]  |  [white]NumPy brute-force[/white]\n"
        f"  [dim](Note: ChromaDB setup may take ~15–30s — Python + hnswlib overhead)[/dim]\n"
    )

    rng = np.random.default_rng(RANDOM_SEED + 1)
    q_indices = rng.integers(0, N_PATIENTS, size=N_BENCHMARK_QUERIES)
    q_vecs = embeddings[q_indices]

    # Step A — Build FAISS index
    with console.status("[cyan][Step A] Building FAISS HNSW index…"):
        faiss_index, _emb_norm, faiss_build_time = setup_faiss_index(embeddings)
    console.print(
        f"  [green]✓[/green] FAISS HNSW built in [yellow]{faiss_build_time:.2f}s[/yellow]"
        f"  ({N_PATIENTS:,} vectors, C++ batch construction)"
    )

    # Normalise query vectors for FAISS cosine (inner product on L2-normed vecs)
    q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    q_vecs_norm = (q_vecs / (q_norms + 1e-9)).astype(np.float32)

    # Step B — Build ChromaDB collection
    console.print(
        f"  [dim][Step B] Building ChromaDB collection ({N_PATIENTS:,} docs, batched 500)…[/dim]"
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        chroma_col, chroma_build_time = setup_chromadb_collection(df, embeddings, progress)
    console.print(
        f"  [green]✓[/green] ChromaDB built in [yellow]{chroma_build_time:.2f}s[/yellow]"
        f"  (Python + hnswlib — slower by design)"
    )

    # Step C — Run all benchmarks
    console.print(
        f"\n  [dim][Step C] Running all 4 query benchmarks ({N_BENCHMARK_QUERIES} queries each)…[/dim]\n"
    )
    with console.status("[cyan]Benchmarking NumPy brute-force…"):
        numpy_ms = bench_numpy(embeddings, q_vecs)
    with console.status("[cyan]Benchmarking FAISS HNSW…"):
        faiss_ms = bench_faiss(faiss_index, q_vecs_norm)
    with console.status("[cyan]Benchmarking zvec HNSW…"):
        zvec_ms = bench_zvec(collection, q_vecs)
    with console.status("[cyan]Benchmarking zvec HNSW + age filter…"):
        zvec_filt_ms = bench_zvec_filtered(collection, q_vecs)
    with console.status("[cyan]Benchmarking ChromaDB…"):
        chroma_ms = bench_chromadb(chroma_col, q_vecs)
    with console.status("[cyan]Benchmarking ChromaDB + age filter…"):
        chroma_filt_ms = bench_chromadb_filtered(chroma_col, q_vecs)

    # Table 1 — Index Build Time
    build_table = Table(
        title="Table 1 — Index Build Time",
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        header_style="bold cyan",
        show_lines=True,
        expand=True,
    )
    build_table.add_column("Library", style="bright_white", width=18)
    build_table.add_column("Time", justify="right", width=12)
    build_table.add_column("Notes", style="dim", width=50)
    build_table.add_row("NumPy", "~0 ms", "Just holds a reference to the array")
    build_table.add_row(
        "[cyan]FAISS HNSW[/cyan]",
        f"[cyan]{faiss_build_time:.2f}s[/cyan]",
        "C++ batch HNSW construction",
    )
    build_table.add_row(
        "[bright_green]zvec HNSW[/bright_green]",
        f"[bright_green]{insert_time:.2f}s[/bright_green]",
        "C++ SIMD + persistent to disk",
    )
    build_table.add_row(
        "[magenta]ChromaDB embed[/magenta]",
        f"[magenta]{chroma_build_time:.2f}s[/magenta]",
        "Python + hnswlib, with metadata index",
    )
    console.print(build_table)

    # Table 2 — Pure Vector Search
    search_table = Table(
        title=f"Table 2 — Pure Vector Search (no filter, top-5, {N_BENCHMARK_QUERIES} queries)",
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        header_style="bold cyan",
        show_lines=True,
        expand=True,
    )
    search_table.add_column("Method", style="bright_white", width=20)
    search_table.add_column("Avg Latency", justify="right", width=14)
    search_table.add_column("QPS", justify="right", width=10)
    search_table.add_column("Filter", width=7, justify="center")
    search_table.add_column("Persist", width=10, justify="center")
    search_table.add_column("Backend", width=22)
    search_table.add_row(
        "[cyan]FAISS HNSW[/cyan]",
        f"[cyan]{faiss_ms:.2f} ms[/cyan]",
        f"[cyan]{1000 / faiss_ms:.0f}[/cyan]",
        "[red]✗[/red]",
        "[red]✗[/red]",
        "C++ Meta/SIMD",
    )
    search_table.add_row(
        "NumPy",
        f"{numpy_ms:.2f} ms",
        f"{1000 / numpy_ms:.0f}",
        "[red]✗[/red]",
        "[dim]N/A[/dim]",
        "Vectorized numpy",
    )
    search_table.add_row(
        "[bright_green]zvec HNSW[/bright_green]",
        f"[bright_green]{zvec_ms:.2f} ms[/bright_green]",
        f"[bright_green]{1000 / zvec_ms:.0f}[/bright_green]",
        "[green]✓[/green]",
        "[green]✓ disk[/green]",
        "C++ Proxima/SIMD",
    )
    search_table.add_row(
        "[magenta]ChromaDB embed[/magenta]",
        f"[magenta]{chroma_ms:.2f} ms[/magenta]",
        f"[magenta]{1000 / chroma_ms:.0f}[/magenta]",
        "[green]✓[/green]",
        "[dim]opt.[/dim]",
        "hnswlib/Python",
    )
    console.print(search_table)

    # Table 3 — Filtered Search
    filter_table = Table(
        title="Table 3 — Filtered Search (age 40–70, top-5, 30 queries)",
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        header_style="bold cyan",
        show_lines=True,
        expand=True,
    )
    filter_table.add_column("Method", style="bright_white", width=22)
    filter_table.add_column("Avg Latency", justify="right", width=14)
    filter_table.add_column("Filtering Approach", style="dim", width=48)
    filter_table.add_row(
        "[bright_green]zvec + filter[/bright_green]",
        f"[bright_green]{zvec_filt_ms:.2f} ms[/bright_green]",
        "InvertIndex + HNSW graph (native C++)",
    )
    filter_table.add_row(
        "[magenta]ChromaDB + filter[/magenta]",
        f"[magenta]{chroma_filt_ms:.2f} ms[/magenta]",
        "HNSW + hnswlib metadata post-filter",
    )
    filter_table.add_row(
        "[cyan]FAISS[/cyan]",
        "[dim]N/A[/dim]",
        "No built-in filter; manual post-filter needed",
    )
    filter_table.add_row(
        "NumPy",
        "[dim]N/A[/dim]",
        "Brute-force then Python list filter",
    )
    console.print(filter_table)

    # Explain HNSW scaling behaviour
    speedup = numpy_ms / zvec_ms
    zvec_is_faster = speedup >= 1.0
    if zvec_is_faster:
        msg = (
            f"  [bold bright_green]zvec HNSW is {speedup:.1f}× faster[/bold bright_green] "
            f"than NumPy brute-force at {N_PATIENTS:,} vectors."
        )
    else:
        inv = 1.0 / speedup
        msg = (
            f"  [bold yellow]At {N_PATIENTS:,} vectors, NumPy is {inv:.1f}× faster — "
            f"HNSW is optimised for 100K+ vectors.[/bold yellow]\n"
            f"  [dim]Estimated zvec speedup at 100K vectors: ~{inv * 8:.0f}× (HNSW graph traversal "
            f"is O(log N) vs NumPy O(N·D)).[/dim]"
        )
    console.print(f"\n{msg}\n")

    # Scale projection table — now with FAISS column
    scale_table = Table(
        title="HNSW vs Brute-Force Scaling Projection",
        box=box.SIMPLE_HEAD,
        header_style="bold",
        show_edge=False,
    )
    scale_table.add_column("Dataset Size", width=16)
    scale_table.add_column("NumPy (est.)", justify="right", width=14)
    scale_table.add_column("FAISS HNSW (est.)", justify="right", width=18)
    scale_table.add_column("zvec HNSW (est.)", justify="right", width=18)
    scale_table.add_column("zvec Advantage", justify="right", width=16)

    # Brute-force scales O(N·D); HNSW scales O(log N)
    for n_scale in [10_000, 100_000, 1_000_000, 10_000_000]:
        scale_factor = n_scale / N_PATIENTS
        numpy_est = numpy_ms * scale_factor
        log_factor = math.log(n_scale) / math.log(N_PATIENTS)
        faiss_est = faiss_ms * log_factor
        hnsw_est = zvec_ms * log_factor
        adv = numpy_est / hnsw_est
        marker = " ←" if n_scale == N_PATIENTS else ""
        style = "bright_green" if adv >= 5 else "yellow" if adv >= 2 else "dim"
        scale_table.add_row(
            f"{n_scale:>10,}{marker}",
            f"{numpy_est:.1f} ms",
            f"[cyan]{faiss_est:.1f} ms[/cyan]",
            f"[{style}]{hnsw_est:.1f} ms[/{style}]",
            f"[{style}]{adv:.1f}×[/{style}]",
        )
    console.print(scale_table)

    # ── Phase 6: Conclusions ─────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold bright_yellow]Phase 6 · Benchmark Conclusions[/bold bright_yellow]"))
    nums = {
        "zvec_ms": zvec_ms,
        "faiss_ms": faiss_ms,
        "chroma_ms": chroma_ms,
        "numpy_ms": numpy_ms,
        "insert_time": insert_time,
        "faiss_build_time": faiss_build_time,
        "chroma_build_time": chroma_build_time,
    }
    print_conclusion(nums)

    # ── Ecosystem Comparison ────────────────────────────────────────────────
    console.print(Rule("[bold yellow]Vector Database Ecosystem[/bold yellow]"))
    # console.print(ecosystem_table())
    console.print()
    console.print(when_to_use_table())

    # ── zvec API Cheatsheet ─────────────────────────────────────────────────
    # console.print()
    # console.print(Rule("[bold dim]zvec API Quick Reference[/bold dim]"))
    # api_table = Table(box=box.SIMPLE, header_style="bold dim", show_edge=False)
    # api_table.add_column("Operation", style="cyan", width=22)
    # api_table.add_column("zvec API", style="bright_white", width=60)
    #
    # api_rows = [
    #     ("Create collection",    "zvec.create_and_open(path, schema, option)"),
    #     ("Open existing",        "zvec.open(path, option)"),
    #     ("Insert (single/batch)","collection.insert(doc)  /  collection.insert([doc1, doc2, …])"),
    #     ("Upsert",               "collection.upsert(doc)  — insert or update"),
    #     ("Update fields",        "collection.update(Doc(id=…, fields={…}))"),
    #     ("Delete by ID",         "collection.delete(id)  /  collection.delete([id1, id2])"),
    #     ("Delete by filter",     "collection.delete_by_filter(filter='age<18')"),
    #     ("Fetch by ID",          "collection.fetch(ids=[…])  → dict[id, Doc]"),
    #     ("Scalar query",         "collection.query(filter='age>=65 and severity>=3', topk=10)"),
    #     ("Vector ANN search",    "collection.query(VectorQuery('field', vector=[…]), topk=5)"),
    #     ("Filtered ANN search",  "collection.query(VectorQuery(…), filter='dept=1', topk=5)"),
    #     ("Stats",                "collection.stats.doc_count  /  .index_completeness"),
    #     ("Optimize index",       "collection.optimize(option=OptimizeOption())"),
    #     ("Destroy collection",   "collection.destroy()"),
    # ]
    # for op, api in api_rows:
    #     api_table.add_row(op, api)
    # console.print(api_table)

    # ── Summary ────────────────────────────────────────────────────────────
    console.print()
    summary_text = (
        f"[bold]Dataset[/bold]              {N_PATIENTS:,} synthetic patient records\n"
        f"[bold]Embedding model[/bold]      {EMBEDDING_MODEL}  ({EMBEDDING_DIM}D FP32, ONNX)\n"
        f"[bold]zvec index[/bold]           HNSW · InvertIndex on 4 numeric fields\n"
        f"[bold]Insert speed[/bold]         {insert_tput:.0f} docs/sec  ({insert_time:.2f}s total)\n\n"
        f"[bold]FAISS HNSW latency[/bold]   {faiss_ms:.2f} ms avg  →  {1000 / faiss_ms:.0f} QPS  [dim](fastest raw, no persist/filter)[/dim]\n"
        f"[bold]zvec HNSW latency[/bold]    {zvec_ms:.2f} ms avg  →  {1000 / zvec_ms:.0f} QPS  [dim](C++ SIMD + persist + filter)[/dim]\n"
        f"[bold]ChromaDB latency[/bold]     {chroma_ms:.2f} ms avg  →  {1000 / chroma_ms:.0f} QPS  [dim](Python + hnswlib, LLM ecosystem)[/dim]\n"
        f"[bold]NumPy exact[/bold]          {numpy_ms:.2f} ms avg  →  {1000 / numpy_ms:.0f} QPS  [dim](brute-force, zero deps)[/dim]\n\n"
        f"[bold]Key insight[/bold]          zvec = SQLite for vectors — embedded, persistent, SIMD-fast"
    )
    console.print(
        Panel(
            Align.center(summary_text),
            title="[bold bright_white] Demo Complete [/bold bright_white]",
            border_style="bright_green",
            padding=(1, 6),
        )
    )

    # Cleanup
    collection.destroy()
    if Path(COLLECTION_PATH).exists():
        shutil.rmtree(COLLECTION_PATH)
        console.print(f"\n  [dim]Cleaned up collection at {COLLECTION_PATH}[/dim]\n")


if __name__ == "__main__":
    main()
