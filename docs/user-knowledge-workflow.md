# User Knowledge Workflow (RAGForge V4)

This document describes the two fundamentally different ways a RAGForge knowledge
base gets its knowledge, and why they are deliberately decoupled from evaluation.

---

## 1. The architectural distinction

```
Knowledge Base creation
        ↓
NO benchmark required
        ↓
BUILD
        ↓
READY

…and, separately, at any time:

Knowledge Base
        ↓
Optional Evaluation Dataset
        ↓
Benchmark (snapshot)
        ↓
Evaluation
```

| Concept | What it is | Required to build a KB? |
|---|---|---|
| **Knowledge Base** | The product artifact: documents → chunks → vectors → retrieval | — |
| **Evaluation Dataset** | Ground-truth questions a human authored | **No** |
| **Benchmark** | An immutable snapshot of reviewed questions | No |
| **Evaluation Run** | Metrics computed against a benchmark | No |

A knowledge base with `evaluation_status = "not_configured"` is a **valid, fully
usable** state. `KBOverview.evaluation_required` is hard-coded `false` and there
is no code path in which a missing benchmark blocks `READY`.

### 1.1 Important limitation: benchmarks are per-domain, hand-made, and rare

**A new knowledge base does not automatically get ground truth.** RAGForge does
not generate verified ground truth for you, and no benchmark ships with a new
knowledge base.

The only benchmark in this repository is the **Automobile Engineering** one
(`benchmarks/automobile-engineering-baseline-v1.json`, 28 questions). It is a
**research artifact specific to that domain**, and it exists because of deliberate
human work: someone read the indexed chunks, identified the passage that answers
each question, recorded the quoted passage as provenance, and then human-reviewed
the whole set before it was frozen.

A brand-new domain — a Naval Architecture knowledge base, say — has **no
benchmark and no measured retrieval quality at all** until someone performs that
same process:

```
author candidate questions  →  human review  →  APPROVED
   →  snapshot               →  human review  →  FROZEN benchmark
   →  run evaluation         →  real Recall@K / MRR / NDCG
```

Until then:

* The KB is fully usable for retrieval — indexing and search do not depend on it.
* Its retrieval quality is **unmeasured**, not "good". The KB overview says
  `Evaluation: Not configured`, and that is the whole truth.
* No metric, percentage or quality claim may be shown for it. RAGForge returns
  `null` for metrics when ground truth is missing, never a placeholder number.

This is deliberate. Auto-generated "ground truth" would be a fabricated
evaluation instrument, which is exactly the failure mode RAGForge exists to avoid.

---

## 2. Source modes

`KnowledgeBase.source_mode` records where knowledge comes from.

| Mode | Meaning | Discovery | User files |
|---|---|---|---|
| `EXTERNAL` | RAGForge discovers and scores authoritative public sources | user-URL + arXiv providers | — |
| `USER_PROVIDED` | The user supplies documents and/or URLs | — | uploads + user URLs |
| `MIXED` | Both | user-URL + arXiv providers | uploads + user URLs |

Mode is metadata: it changes which UI affordances are offered and which quality
model scores a source. It never gates evaluation and never changes the pipeline.

### Backwards compatibility

`source_mode` defaults to `EXTERNAL`, so every pre-V4 knowledge base reads back
correctly and behaves exactly as before. `version` defaults to `1` and
`last_build_at` to `null`.

---

## 3. Building a knowledge base from your own documents

The primary product path. A Naval Architecture student can upload lecture decks,
professor PDFs, textbooks, notes, lab manuals and past question papers, and get a
searchable, provenance-carrying knowledge base without pasting anything into a
chat window.

### Step 1 — Create

`POST /api/knowledge-bases` with `source_mode: "user_provided"`. Nothing else is
required. The UI (`/knowledge-bases/new`) walks: **Step 1 Domain → Step 2
Sources → Step 3 Ingest**.

### Step 2 — Upload

`POST /api/knowledge-bases/{kb_id}/documents/upload` (multipart, `files[]`).

Every file passes through `validate_upload()` **before** anything touches disk:

1. **Name safety** — directory components (`../../etc/passwd`, `C:\Users\…`) are
   stripped; unsafe characters replaced; length capped at 180 characters.
2. **Size** — `MAX_UPLOAD_BYTES` (default 100 MB) and
   `MAX_UPLOAD_FILES_PER_REQUEST` (default 50).
3. **Extension** — must be one of `supported_extensions()`.
4. **Magic bytes** — `.pdf` must start with `%PDF`; `.pptx`/`.docx` must be a ZIP
   (`PK\x03\x04`); `.ppt` must be an OLE2 container.
5. **Container integrity** — OOXML files are opened as ZIPs and must contain the
   part their extension promises (`ppt/presentation.xml`, `word/document.xml`).
   A `.docx` renamed to `.pptx` is rejected, not silently mis-parsed.

Rejections are **per file** and never abort the batch. The response reports
`uploaded / duplicates / rejected / failed` plus a per-file message.

### Step 3 — Parse

| Format | Parser | Structural unit | Provenance emitted |
|---|---|---|---|
| PDF | `PdfParser` (pypdf) | page | `[PAGE n]` → `chunk.page` |
| PPTX | `PptxParser` (python-pptx) | slide | `[SLIDE n]` + slide title → `chunk.slide`, `chunk.slide_title` |
| DOCX | `DocxParser` (python-docx) | heading | `# Heading` → `chunk.section`, `chunk.section_path`; tables kept as pipe rows |
| HTML | `HtmlParser` (BeautifulSoup) | — | cleaned text |
| Markdown | `MarkdownParser` | heading | heading structure preserved |
| TXT | `TextParser` | — | normalized text |
| PPT (legacy) | `LegacyPptParser` | — | explicit rejection: "Re-save the deck as .pptx" |

**Numbers are never invented.** A deck with no title placeholder yields
`slide_title = None`, not a synthesised "Slide 7". A PDF with no extractable
text raises an explicit error naming OCR as the reason — OCR is not implemented.

DOCX tables are rendered as pipe rows (`| Cell | Cell |`) rather than flattened
into prose, so structured content survives into chunks.

### Step 4 — Index

Upload accepts `index=true` to parse → chunk → embed → index immediately, or the
Documents page indexes a single document on demand.

---

## 4. Source quality for user-provided files

The external `SourceQualityScorer` measures *publication* signals: domain
authority, HTTP accessibility, publication date, evidence markers. None of those
exist for a lecture PDF a student uploaded at 2am. Applying it would systematically
reject exactly the material the user cares about.

`UserProvidedIntegrityScorer` (`app/services/source_quality/user_scorer.py`)
therefore uses **only** integrity signals:

| Signal | Weight | Meaning |
|---|---|---|
| `file_validity` | 0.30 | readable, not corrupt, parser opened it |
| `content_extraction` | 0.25 | meaningful text was extracted |
| `user_relevance` | 0.20 | relevance asserted by the uploader |
| `duplication` | 0.15 | no identical content already present |
| `structure` | 0.10 | pages / slides / headings recovered |

There is **no** authority, recency, accessibility or evidence weight. The
web-signal fields stay at a neutral `0.5` ("not applicable"), never `0.0`.

**Decisions**

| Condition | Decision |
|---|---|
| File failed validation | `REJECT` |
| No text extractable at all | `REJECT` |
| Validated but the parser could not read it | `REVIEW` |
| Thin content (< 200 chars) | `ACCEPT` + warning |
| Duplicate content | `ACCEPT` + warning (upload short-circuits it first) |

Thin content is **never** a rejection reason — a one-paragraph class note is a
legitimate document.

---

## 5. Document library

`GET /api/knowledge-bases/{kb_id}/documents` and `GET .../document-library`.

Status lifecycle:

```
UPLOADED → PARSING → PARSED → CHUNKING → INDEXING → READY
                   ↘ FAILED
```

Operations:

| Action | Endpoint | Effect |
|---|---|---|
| Inspect | `GET /documents/{id}` | document + source + integrity + chunk count |
| Read text | `GET /documents/{id}/text` | normalized extracted text, with honest truncation |
| View chunks | `GET /documents/{id}/chunks` | chunks with full provenance |
| Rebuild | `POST /documents/{id}/rebuild` | re-parse from the stored original, optionally re-index |
| Re-index | `POST /documents/{id}/index` | chunk → embed → index **only this document** |
| Replace | `POST /documents/{id}/replace` | new file becomes a new `document_version` |
| Remove | `DELETE /documents/{id}` | removes document + chunks **+ vectors** |

Deleting a document removes its vectors. Deleting only the database rows would
leave the chunks permanently retrievable from Qdrant.

---

## 6. Incremental ingestion

Adding the 101st document to a 100-document knowledge base must not re-embed the
other 100. Both the full build (`POST /{kb_id}/index`) and the incremental path
(`POST /documents/{id}/index`) call one shared function,
`app/services/indexing/document_indexer.py::index_documents`, so they cannot drift
apart in how stale vectors are handled.

```
read parsed text → chunk → ensure_collection
  → delete_document_vectors(only these documents)   ← BEFORE upsert
  → embed → upsert
  → replace chunk rows (only after a successful upsert)
  → [full build only] sweep orphaned points
```

Why the order matters:

* Re-chunking always generates **new chunk IDs**. Upserting alone would leave the
  old points orphaned and retrievable forever.
* Chunk rows are written **after** a successful upsert, so SQLite always holds
  exactly the chunks present in the vector store.
* An incremental run deliberately does **not** sweep orphans: it must not pay to
  scan the whole collection and must never touch vectors belonging to documents
  it did not process.
* If a backend cannot delete vectors, indexing **refuses** rather than corrupt.

### Replacement

`POST /documents/{id}/replace`:

1. Rejects byte-identical uploads (409) and content that already exists elsewhere
   in the knowledge base (409).
2. Ingests the new file as a **new document** with `document_version = old + 1`
   and `replaces_document_id = old.id`.
3. **Deletes the old document's vectors.**
4. Marks the old document `FAILED` with `parse_error = "Superseded by …"` and
   keeps it as history.

Historical experiment artifacts are never destroyed.

### Versioning

* `Document.document_version` — per-document file revision.
* `KnowledgeBase.version` — incremented on every successful index.
* `Chunk.kb_version` / `Chunk.document_version` — stamped on every chunk, so a
  retrieved chunk can be traced back to the exact index that produced it.

---

## 7. Provenance

Every chunk carries:

```
document_id, source_id, content_hash, chunk_id, section_path,
page, slide, slide_title, chunk_index, chunking_strategy,
chunking_config, document_version, user_provided, kb_version
```

The retrieval response exposes all of it, plus enough for a future LLM layer to
cite precisely:

```json
{
  "chunk_id": "chk_1a2b3c4d5e6f",
  "document_id": "doc_9f8e7d6c5b4a",
  "text": "Free surface effect reduces the effective metacentric height…",
  "score": 0.84,
  "provenance": {
    "kb_id": "kb_1a2b3c4d5e6f",
    "document_title": "NA5010_Lecture_08.pptx",
    "document_version": 2,
    "slide": 17,
    "slide_title": "Free Surface Effect",
    "section_path": "Free Surface Effect",
    "source_id": "src_0a1b2c3d4e5f",
    "user_provided": true,
    "chunking_strategy": "section-aware",
    "kb_version": 3
  }
}
```

Absent values are **omitted, not zero-filled** — a TXT chunk has no `page` key at
all. The Retrieval Lab renders this as a chain:

```
Knowledge Base → Document → Source → Slide/Page/Section → Chunk
```

---

## 8. Privacy

User-provided files may be private course material.

* Parsing, chunking and embedding are **local** by default
  (SentenceTransformers).
* `ingest_uploaded_bytes` performs **no network access at all** — enforced by a
  test that makes `httpx.Client.get` raise.
* `Settings.allow_external_llm_for_user_documents` defaults to `False` and exists
  to make the decision explicit rather than implicit.
* User URLs are still SSRF-validated and accessibility-probed before ingestion.

---

## 9. The three workflows side by side

| | External KB | User-provided KB | Mixed KB |
|---|---|---|---|
| Sources | discovered + scored | uploads / user URLs | both |
| Quality model | authority, recency, evidence, accessibility | file integrity | both models, per source |
| Source review | full (authority, relevance, score, decision) | lightweight (validity, parser, size, duplicate) | both |
| Parser | PDF/HTML/TXT/MD via download | PDF/PPTX/DOCX/TXT/MD/HTML | both |
| Needs domain spec | yes | no | yes |
| Needs benchmark | **no** | **no** | **no** |

---

## 10. Current limitations

* **No benchmark for a new domain.** See §1.1. Retrieval quality for a new KB is
  unmeasured until a human authors, reviews and freezes a benchmark for it.
* No OCR — scanned/image-only PDFs and images fail with an explicit message.
* No video/YouTube transcripts, CSV, XLSX or image formats.
* Legacy binary `.ppt` is rejected (needs LibreOffice to convert).
* Embedding is synchronous: a 500-document upload blocks until it finishes. There
  is no job queue.
* Speaker notes are captured for PPTX but have no dedicated provenance field.
* `parse_metadata` is preserved but not exposed in the UI.
* No authentication — the tool is a single-user local research instrument.
* Search is dense-only. BM25 / hybrid / reranking are architectural extension
  points (`RETRIEVERS` registry) but are **not implemented** and are not
  advertised.