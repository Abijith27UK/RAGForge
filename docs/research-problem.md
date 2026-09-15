# RAGForge — Research Problem

## Why RAG quality depends on knowledge-base construction

Most RAG systems treat the knowledge base as a given: dump documents in, chunk, embed, retrieve. But every downstream failure mode traces back to construction decisions:

- **Irrelevant or low-authority sources** → the retriever faithfully returns bad evidence. No amount of retrieval tuning fixes a corpus that doesn't contain the right knowledge.
- **Naive chunking** → sections split mid-argument, tables destroyed, context lost; the embedding of a broken fragment doesn't represent any meaningful unit of knowledge.
- **Missing provenance** → answers can't be traced to an authority, so users can't verify or trust them.
- **No evaluation** → nobody knows whether the corpus covers the domain or whether retrieval works, until the system fails in production use.

The quality of a RAG system is therefore bounded *at construction time*, not at query time.

## The five sub-problems RAGForge targets

1. **Source selection problem** — Given a domain, which sources *should* enter the corpus?
   Authority, relevance, recency, accessibility, duplication, and evidence quality all
   matter, and the decision must be *explainable and overridable*, not an opaque score.

2. **Chunking problem** — How should documents be segmented so that chunks are meaningful
   units of domain knowledge? Structure-aware splitting preserves the author's own
   hierarchy (sections, headings, pages) instead of arbitrary 512-token windows.

3. **Retrieval problem** — How should the corpus be searched so the right knowledge unit
   is returned? Dense retrieval is the baseline; the architecture reserves room for
   keyword, hybrid, and reranking strategies that can be compared experimentally.

4. **Evaluation problem** — How do we know it works? Retrieval metrics (Recall@K,
   Precision@K, MRR, NDCG) computed against explicit ground truth, with missing ground
   truth reported as unknown rather than fabricated.

5. **Optimization problem** — Given measured weaknesses, which construction decisions
   should change? This requires that every pipeline choice (chunker, embeddings, source
   set, retrieval strategy) is a *configurable, swappable component*.

## How this differs from a normal RAG chatbot

| Generic RAG chatbot            | RAGForge |
|--------------------------------|----------|
| Corpus is user-dumped          | Corpus is *constructed*: discovered, scored, curated |
| Chunking is a config checkbox  | Chunking is a first-class, comparable strategy |
| "Quality" is a vibe or a fake %| Quality assessments carry signals, reasons, warnings |
| Provenance optional            | Provenance is mandatory on every chunk |
| Evaluation rarely exists       | Evaluation is the research instrument |
| One fixed pipeline             | Pipeline stages are interfaces, built for comparison |

## Positioning against existing platforms

Dify, LangChain-based stacks, and similar platforms already provide ingestion, chunking,
embedding, vector storage, and RAG workflows. RAGForge does **not** claim novelty for any
of those primitives. The research contribution is the **automated construction loop**:

```
domain specification → knowledge requirements → source discovery →
source quality/inclusion decisions → ingestion → domain-aware chunking →
retrieval → evaluation → optimization
```

and the ability to *measure* whether that loop produces better knowledge bases than
indiscriminate construction (the baseline-vs-RAGForge experiments).

## Research hypotheses (to be tested with the experiment infrastructure)

- **H1**: Quality-filtered sources yield higher retrieval metrics than indiscriminate
  sources at equal corpus size. (Experiment 1)
- **H2**: Structure-aware chunking beats fixed-size chunking on section-boundary
  questions. (Experiment 2)
- **H3**: Hybrid retrieval beats dense-only when the domain has strong terminology.
  (Experiment 3 — not yet implemented)
- **H4**: Reranking improves precision without hurting recall. (Experiment 4 — not yet implemented)
- **H5**: Configuration optimized via evaluation feedback outperforms manual defaults. (Experiment 5)
