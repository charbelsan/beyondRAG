"""
ingestion/loader.py
───────────────────
Loads PDFs, DOCX, and TXT files from `docs/` and yields LangChain
Document objects.  Splitting mode is configured in configs/pipeline.yaml.
"""

from __future__ import annotations

import uuid, yaml, pathlib, logging, docx
import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CFG = yaml.safe_load(pathlib.Path("configs/pipeline.yaml").read_text())
CH  = CFG["chunking"]

# ── splitters ───────────────────────────────────────────────────────────────
def _splitter():
    if CH["mode"] == "semantic":
        emb = HuggingFaceEmbeddings(
            model_name=CH["semantic"]["embeddings_model"]
        )
        return SemanticChunker(
            embeddings=emb,
            breakpoint_threshold_type  = CH["semantic"].get("breakpoint_threshold_type", "percentile"),
            breakpoint_threshold_amount = CH["semantic"].get("breakpoint_threshold_amount", 0.6),
        )
    if CH["mode"] == "recursive":
        rc = CH["recursive"]
        return RecursiveCharacterTextSplitter(
            chunk_size   = rc["chunk_size"],
            chunk_overlap= rc["chunk_overlap"],
        )
    return None

_SPLIT = _splitter()

def _split(text: str) -> list[str]:
    return [text] if CH["mode"] == "page" else [d.page_content for d in _SPLIT.create_documents([text])]

# ── helpers for each file type ──────────────────────────────────────────────
def _yield_pdf(p: pathlib.Path):
    for pg in pymupdf4llm.to_markdown(p, page_chunks=True):
        # normalise key to page_number
        page_num = pg["metadata"].get("page_number") or pg["metadata"].get("page")
        if page_num is None:
            logging.warning("Skip %s: page metadata missing", p)
            continue

        base_meta = {
            "source": str(p),
            "page_number": page_num,
        }

        for ch in _split(pg["text"]):
            meta = base_meta.copy()
            meta["doc_id"]   = f"{uuid.uuid4()}_{page_num}"
            meta["raw_text"] = ch                                 # ← NEW
            yield Document(page_content=ch, metadata=meta)

def _yield_txt(t: str, src: str):
    for ch in _split(t):
        yield Document(
            page_content=ch,
            metadata={
                "source": src,
                "doc_id": str(uuid.uuid4()),
                "raw_text": ch,                                   # ← NEW
            },
        )

# ── public loader -----------------------------------------------------------
def load_documents(root: str = "docs"):
    for p in pathlib.Path(root).rglob("*"):
        if not p.is_file():
            continue
        try:
            suf = p.suffix.lower()
            if suf == ".pdf":
                yield from _yield_pdf(p)
            elif suf == ".docx":
                doc = docx.Document(p)
                yield from _yield_txt("\n".join(x.text for x in doc.paragraphs), str(p))
            elif suf == ".txt":
                yield from _yield_txt(p.read_text(errors="ignore"), str(p))
        except Exception as e:
            logging.warning("Skip %s: %s", p, e)
