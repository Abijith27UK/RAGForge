# V21 — Second-Domain Real-LLM Validation: Execution Plan

Status: **plan written BEFORE execution** (2026-10-19). Results are appended in
a separate section only after the run, so scope, corpus, acceptance criteria and
evaluation methodology are declared up front and cannot be quietly adjusted
after the numbers are known.

## 0. Relationship to V19

The corpus, question set, chunking configuration and acceptance criteria below
are **inherited unchanged from the V19 pre-registration**
([`v19-second-domain-validation.md`](./v19-second-domain-validation.md)), which
was written before any corpus was fetched and could not be executed because no
real LLM provider was constructible (`gemini` was a declared-but-unimplemented
provider name; the configured key was exposed and unusable). V20 implemented
`LLM_PROVIDER=gemini` through the existing OpenAI-compatible surface and the
operator has since run `scripts/v20_verify_provider.py --live` successfully:

| Evidence (operator-supplied, 2026-10-19) | Value |
|---|---|
| `LLM_PROVIDER` | `gemini` |
| `LLM_MODEL` | `gemini-3.8-flash` |
| Provider type | `OpenAICompatibleProvider` |
| `provider_is_mock` | `False` |
| Endpoint | Google OpenAI-compatible Gemini endpoint |
| One structured-output call | succeeded, 4.43 s, schema validation passed |

That output is treated as the provider smoke test; V21 does not repeat it
unnecessarily. Every LLM-backed operation *inside the workflow itself* must
still record its own `is_mock` flag — the smoke test proves the provider
constructs and answers, not that each pipeline step used it.

V19's Phase-8 section records what was NOT executed in October. This document
owns what V21 executes.

## 1. Experiment scope

**Domain:** Internet Protocols / Network Engineering — a course-reference KB
for network-engineering students: protocol behaviour, field semantics,
parameter values and terminology from primary IETF standards, answered with
citations.

**Question:** does the full construction loop (domain specification → source
discovery → source-quality decisions → human-approved ingestion → chunking →
embedding → index → retrieval → grounded answering → abstention) produce a
usable, provenance-rich KB in a **materially different domain** from Automobile
Engineering, with **real** LLM calls honestly recorded at every step?

**Out of scope (unchanged from V19):** vendor configuration guides, non-IETF
protocols, physical/wireless layers, RFCs not in the corpus list, tuning
chunking/retrieval for this domain, any claim of statistical significance,
human-approval or freeze of any benchmark artifact, and any modification of the
protected Automobile KB or frozen benchmarks.

## 2. Corpus selection criteria and decision

Criteria (declared before fetch): bounded size; single authoritative publisher;
immutable, numbered primary documents; plain-text parseable by the existing
`TextParser`; freely redistributable; retrieval-useful (terminology-dense,
verifiable claims); materially different source *type* from the Automobile
corpus (lecture notes/web pages → primary standards).

**Decision: the RFC corpus is appropriate.** All 14 URLs were re-verified
HTTP 200 today (sizes below). Every source is the RFC Editor's own rendering
of a published RFC. Authority derives from publisher and document type, not
from an LLM opinion. Copyright: IETF Trust / BCP 78 — verbatim reproduction
and quotation for reference is expressly permitted (verified in an RFC
copyright notice). Accessibility: plain `.txt`, no auth, no robots issues.
The quality engine still scores each URL independently and a REVIEW decision
remains a human approval gate.

| # | Subdomain | RFC | ~bytes (measured today) |
|---|---|---|---|
| 1 | ipv4 | 791 | 94,892 |
| 2 | icmp | 792 | 29,186 |
| 3 | link-layer | 826 | 21,556 |
| 4 | addressing | 1918 | 22,271 |
| 5 | addressing | 5737 | 7,036 |
| 6 | ipv6 | 8200 | 93,162 |
| 7 | transport | 9293 | 263,696 |
| 8 | dns | 1034 | 126,103 |
| 9 | dns | 1035 | 122,549 |
| 10 | dhcp | 2131 | 113,738 |
| 11 | routing | 4271 | 222,702 |
| 12 | routing | 2328 | 524,985 |
| 13 | security | 8446 | 337,736 |
| 14 | http | 9110 | 502,941 |

Total ≈ 2.48 MB fetched, 14/14 reachable 2026-10-19.

## 3. Expected resource usage (declared before the run)

* 1 new KB row, 1 new Qdrant collection (`kb_<new_id>`), ≤ 14 documents,
  expected ≈ 1 700–2 400 chunks (V19 estimate; RFC plain text falls back to
  fixed windows inside `section-aware`, recorded per chunk).
* Embedding: local `all-MiniLM-L6-v2` on CPU — minutes, not hours; no API cost.
* Real LLM calls expected: **1** `analyze_domain` structured call +
  **2** in-corpus answer generations (the 3 unsupported probes should be
  refused by the evidence gate *before* generation, so normally 0 calls; if a
  probe reaches generation that is recorded, not hidden) + **0** during
  discovery/ingest/build (source scoring, parsing, chunking, embedding are
  deterministic/local). Total budget ≈ 3–6 real calls; the run stops the phase
  on the first authentication/model/schema failure rather than retrying
  expensively.
* Disk: ~5 MB documents under `DATA_DIR`; no scratch DATA_DIR needed for the
  *live* workflow itself (it must use the real workspace to be the real
  pipeline); the **test suite** runs with an explicit scratch `DATA_DIR`.

## 4. Execution procedure (the required MCP workflow)

Executed by the existing `scripts/v19_second_domain_validation.py` over the
real MCP stdio server — no new orchestration code:

1. `discover` → `analyze_domain` (records `generated_by`, `is_mock`),
   `plan_knowledge_base` → gate refusals (no confirmation; plan-id-as-confirmation)
   → `confirm_plan` → `create_knowledge_base` (replay refusal) →
   `discover_sources` (user-url provider, 14 URLs, real probes + scoring,
   nothing downloaded) → `assess_sources`. State written to
   `backend/data/v19-validation-state.json`.
2. **Human approval pause.** The script stops. The operator reviews the scored
   candidates (ACCEPT/REVIEW/REJECT with reasons) and approves the exact URL
   list via `--approve-url` (repeat per URL). REVIEW-band sources are recorded
   as human-accepted with engine decision and score preserved in the note.
3. `ingest` → operator decision recorded through the same code path as the API
   → `plan_knowledge_base` → `confirm_plan` → `ingest_sources` (real download,
   parse, hash; failures reported) → `plan_knowledge_base` → `confirm_plan` →
   `build_knowledge_base` (chunk, embed, index) → `validate_knowledge_base`
   → 8 retrieval probes (one per subdomain) → `get_evidence` provenance sample
   → 2 in-corpus + 3 unsupported `answer_from_knowledge_base` probes →
   `compare_retrieval_strategies` on the draft instrument (only after the
   benchmark exists; see §5) → `get_operation_status` audit.
4. `audit` → protection checks.

A separate KB is created; `kb_f278c283c748` is measured before/after every
subcommand and the run **stops** if it is not at exactly 812 points.

## 5. Evaluation plan (declared before the run)

1. **Benchmark candidate:** author `benchmarks/internet-protocols-draft-v1.json`
   with the existing `scripts/v19_benchmark_tool.py` (question specs are
   pre-declared *in that file*, versioned with the code): `draft` inspects
   grounding candidates, `finalize` writes the artifact only if every supported
   question's anchor phrase is verified present in a real chunk of the declared
   RFC, `verify` re-checks. Artifact carries `tier=draft`,
   `human_review=pending`, `labels_are_machine_generated=true`. It stays
   DRAFT/PENDING; no freeze, no human-review claim. Unsupported questions carry
   no expected ids (classified UNKNOWN, never scored 0).
2. **Retrieval metrics** via existing runners (`scripts/v19_measurements.py`):
   dense, bm25, hybrid at k=10 with depth-20 probe: recall@5/@10, MRR,
   nDCG@10, precision@10 (lower-bound caveat), per-question latency, evidence
   ranks, failure codes, evidence-availability coverage (expected chunk ids
   present in corpus AND index). Two pre-declared experiments (E1 window k5→k10,
   primary recall; E2 strategy dense→bm25/hybrid, primary MRR) with hypotheses
   and rejection conditions already written in the script. Exploratory only.
3. **Grounded answering:** in-corpus probes must produce citations whose text
   actually supports the answer (manual inspection of a sample); unsupported
   probes must abstain with 0 citations/0 claims. **Answer correctness,
   citation support and abstention quality remain UNKNOWN** — no reference
   answers exist and no human review has happened.
4. **Off-domain gate:** at least one out-of-scope query (e.g. Jupiter-moons
   class) against the NEW KB must abstain. The relevance-floor threshold
   (`ANSWER_MIN_RELEVANCE=0.35`) is calibrated for `all-MiniLM-L6-v2` **on the
   Automobile corpus**; cosine distributions on this corpus are different, so
   the calibration is re-measured here and any false abstention/false answer is
   reported as a calibration limitation, not tuned away.
5. **Claim discipline:** exploratory validation on one small corpus. No
   significance claims; no tune-on-the-evaluated-set; DRAFT ≠ ground truth;
   mock vs real provider recorded per operation.

### Acceptance criteria (inherited from V19, unchanged)

* A1 — workflow completes with a recorded confirmation before every
  consequential step; gate refusals still refuse.
* A2 — all approved sources ingested or explicitly reported failed;
  `validate_knowledge_base` → READY, chunk rows == vector points, integrity
  HEALTHY.
* A3 — ≥ 90 % of labelled questions' expected chunks present in corpus+index.
* A4 — dense recall@10 ≥ 0.70 on the draft set (miss = finding, not tuning).
* A5 — ≥ 6/7 unsupported questions abstain or explicit insufficient-evidence
  (incl. the off-domain probe). *(V19 wrote 6–7 unsupported probes; the
  pre-declared set in the script has 3 + the retrieval probes; V21 adds the
  off-domain gate probe. If fewer than 7 unsupported probes exist, A5 is
  evaluated on the probes that exist and the shortfall is stated, not
  papered over.)*
* A6 — every LLM-backed step records provider identity; mock usage, if any, is
  disclosed and no real-LLM claim is made for it.
* A7 — protection: protected KB stays at exactly 812 points with unchanged
  `updated_at`; SQLite integrity ok; frozen benchmark files byte-identical;
  zero `mcp_operations` rows referencing the protected KB; cleanup inventoried.

## 6. Safety and data integrity (declared before the run)

* Fresh before-state snapshot taken 2026-10-19 (SQLite integrity ok,
  fk_check clean, 23 KB rows, protected KB 11 docs / 812 chunks / 812 points /
  `updated_at=2026-09-15T10:58:31.305933Z`, Qdrant total 8,822 points across
  14 collections, `mcp_operations` 45 total / 32 inert empty-`kb_id` rows,
  sha256 of all 16 benchmark files recorded in the run log).
* Tests and nested subprocesses use an explicit scratch `DATA_DIR`; the V20
  session guard fails any run that writes table counts to the real database.
* The 32 inert historical `mcp_operations` rows are not modified or deleted.
* The untracked held-out benchmark `benchmarks/automobile-engineering-heldout-v1.json`
  and all frozen artifacts are never written. Never committed: `.env`, keys,
  `next-dev.log`, `tsbuildinfo`, generated data.
* Only scratch resources created by this run are deleted; the new KB and its
  collection are **kept** (the draft benchmark is bound to them; deleting them
  would orphan the instrument) and inventoried explicitly.

## 7. ChatGPT MCP integration (Phase 5 scope)

Research completed before design (sources in the final report): ChatGPT's
documented custom-MCP path is **ChatGPT web → Plugins → + → Add custom MCP
server**, requiring a **public HTTPS Streamable-HTTP** endpoint, or OpenAI's
**Secure MCP Tunnel** (which can launch a local **stdio** server without a
public listener). Direct local-stdio in the desktop app is not documented.
Custom connectors are gated: Pro = read/fetch; Business/Enterprise/Edu = full;
no documented Free/Plus entitlement. Auth: anonymous mode, or OAuth 2.1 per
the MCP authorization spec; **ChatGPT cannot present a static API key**.

Therefore the minimum implementation for RAGForge is: (a) a Streamable-HTTP
transport wrapping the **existing** MCP server and tool definitions (no second
retrieval/answering stack), behind explicit auth + TLS + user scoping + rate
limits, **or** (b) OpenAI's tunnel client pointed at the existing stdio
command. Neither endpoint will be exposed unauthenticated. Host integration is
claimed **UNVERIFIED** until an actual ChatGPT conversation completes a tool
call; account capability must be confirmed by the operator first.

## 8. Phase 6 — tests and delivery

Focused: `test_v19_second_domain.py`, `test_v18_mcp_agentic.py`,
`test_v17_mcp_server.py`, plus any test touching changed services; then the
full backend suite with scratch `DATA_DIR`. Frontend checks only if frontend
source changes. Final diff review + secret scan (counts and filenames only).
Commit and push to `origin/Dev_1_midterm` (no force-push), verify remote SHA
with `git ls-remote`.
