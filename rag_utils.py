import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import faiss
import numpy as np
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer


@dataclass
class ChunkRecord:
    chunk_id: int
    file_name: str
    page_num: int
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(end - overlap, 0)

    return chunks


def extract_pdf_pages(uploaded_file) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []

    with pdfplumber.open(uploaded_file) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = normalize_text(page_text)
            if page_text:
                pages.append((idx, page_text))

    return pages


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def process_uploaded_pdfs(
    uploaded_files,
    embed_model: SentenceTransformer,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
):
    chunk_records: List[ChunkRecord] = []
    warnings: List[str] = []
    chunk_id = 0

    for uploaded_file in uploaded_files:
        try:
            pages = extract_pdf_pages(uploaded_file)

            if not pages:
                warnings.append(f"No readable text found in {uploaded_file.name}")
                continue

            for page_num, page_text in pages:
                for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=chunk_overlap):
                    chunk_records.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            file_name=uploaded_file.name,
                            page_num=page_num,
                            text=chunk,
                        )
                    )
                    chunk_id += 1

        except Exception as e:
            warnings.append(f"Failed to read {uploaded_file.name}: {e}")

    if not chunk_records:
        return [], None, warnings

    texts = [record.text for record in chunk_records]
    embeddings = embed_texts(embed_model, texts)
    index = build_faiss_index(embeddings)

    return chunk_records, index, warnings


def retrieve_chunks(
    query: str,
    embed_model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    chunk_records: List[ChunkRecord],
    top_k: int = 8,
):
    query_embedding = embed_texts(embed_model, [query])
    distances, indices = index.search(query_embedding, min(top_k, len(chunk_records)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append((chunk_records[idx], float(dist)))

    return results


def build_context(retrieved, max_chars: int = 14000) -> str:
    parts: List[str] = []
    total_chars = 0

    for rank, (record, dist) in enumerate(retrieved, start=1):
        block = (
            f"[Source {rank}] "
            f"File: {record.file_name} | Page: {record.page_num} | Distance: {dist:.4f}\n"
            f"{record.text}\n"
        )

        if total_chars + len(block) > max_chars:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts)


def group_sources_for_display(retrieved) -> List[Dict]:
    grouped: List[Dict] = []

    for rank, (record, dist) in enumerate(retrieved, start=1):
        grouped.append(
            {
                "rank": rank,
                "file": record.file_name,
                "page": record.page_num,
                "distance": round(dist, 4),
                "excerpt": record.text[:500] + ("..." if len(record.text) > 500 else ""),
            }
        )

    return grouped


def fingerprints(uploaded_files) -> List[Tuple[str, int]]:
    return [(f.name, getattr(f, "size", 0)) for f in uploaded_files]