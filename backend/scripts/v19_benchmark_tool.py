"""V19 — author and verify the DRAFT retrieval benchmark for the second domain.

The benchmark is machine-drafted from the built corpus and explicitly marked
`tier=draft`, `human_review=pending`: nothing here is a human approval, a
frozen instrument or a relevance judgment made by a reviewer. Its only job is
to make the retrieval measurements honest:

* every supported question is grounded in AT LEAST ONE REAL CHUNK whose text
  contains the declared anchor phrase, inside the declared RFC document;
* the grounding is verified again at finalize time and by `verify` afterwards,
  so a label can never point at a chunk that does not carry the evidence;
* unsupported questions carry NO expected ids — the instrument classifies them
  UNKNOWN rather than scoring a fabricated 0 — and are listed separately.

The question spec lives in this file (versioned with the code), each entry
naming the RFC document, an anchor phrase that must appear in the grounding
chunk, expected terminology and a provenance note. This is the same
pre-declaration discipline as scripts/run_retrieval_experiment.py.

V21 note (2026-10-19): the first `draft` pass against the built corpus showed
25/26 anchors were NOT unique — RFC plain text is ~72-column wrapped, so many
obvious phrases either crossed a line break (NOT FOUND) or repeated across the
document (AMBIGUOUS). The anchors and their expected_keywords were then
adjusted to uniquely identify the section-level chunk that carries the answer
evidence. Question text was NOT changed, and no adjustment was made by
observing retrieval results: this is grounding calibration, disclosed here,
not tuning.

Usage (from backend/):
    py -m scripts.v19_benchmark_tool draft    --kb <kb_id>     # inspect grounding candidates
    py -m scripts.v19_benchmark_tool finalize --kb <kb_id>     # write the draft artifact
    py -m scripts.v19_benchmark_tool verify   --kb <kb_id>     # re-verify a written artifact
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

REPO_ROOT = BACKEND_ROOT.parent
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "internet-protocols-draft-v1.json"

#: Authorship block written into the artifact. The tier and review state are
#: what every lifecycle gate reads: draft/pending => exploratory only.
AUTHORSHIP = {
    "tier": "draft",
    "human_review": "pending",
    "drafted_by": (
        "RAGForge V19 agent workflow: machine-drafted from the built corpus and "
        "grounded by exact-phrase verification. NOT human-reviewed."
    ),
    "drafted_at": "",  # filled at finalize
    "review_required_before": (
        "any confirmatory claim, benchmark freeze, official experiment, or use as "
        "a held-out instrument"
    ),
    "labels_are_machine_generated": True,
}

RFC = "https://www.rfc-editor.org/rfc/"

#: The pre-declared question set. `anchor` is a phrase that MUST appear in the
#: grounding chunk; `document_url` restricts grounding to that RFC; unsupported
#: questions declare `unsupported: true` and carry no expected ids by design.
QUESTIONS: list[dict[str, Any]] = [
    # -- ipv4 -----------------------------------------------------------------
    {
        "id": "proto-001",
        "subdomain": "ipv4",
        "question": "What does an IPv4 router do to the Time to Live field of a datagram it forwards?",
        "document_url": RFC + "rfc791.txt",
        "anchor": "decremented",
        "expected_keywords": ["time to live", "decremented", "seconds"],
        "provenance": "RFC 791, Time to Live field semantics (datagram processing at gateways)",
    },
    {
        "id": "proto-002",
        "subdomain": "ipv4",
        "question": "Which unit carries a datagram from one internet module to another in the IPv4 header?",
        "document_url": RFC + "rfc791.txt",
        "anchor": "The internet header field Internet Header Length is the length",
        "expected_keywords": ["internet header length", "header"],
        "provenance": "RFC 791, header format overview (IHL and version fields)",
    },
    # -- icmp -----------------------------------------------------------------
    {
        "id": "proto-003",
        "subdomain": "icmp",
        "question": "Which ICMP message does a gateway send when a datagram's lifetime expires?",
        "document_url": RFC + "rfc792.txt",
        "anchor": "is zero it must discard the datagram. The gateway may also notify",
        "expected_keywords": ["time exceeded", "discard", "gateway"],
        "provenance": "RFC 792, Time Exceeded message definition",
    },
    {
        "id": "proto-004",
        "subdomain": "icmp",
        "question": "How does a host test whether a destination is reachable using ICMP?",
        "document_url": RFC + "rfc792.txt",
        "anchor": "Echo or Echo Reply Message",
        "expected_keywords": ["echo", "destination", "reply"],
        "provenance": "RFC 792, Echo Request/Reply message pair",
    },
    # -- addressing -----------------------------------------------------------
    {
        "id": "proto-005",
        "subdomain": "addressing",
        "question": "Which IPv4 address blocks are reserved for private internets?",
        "document_url": RFC + "rfc1918.txt",
        "anchor": "172.16.",
        "expected_keywords": ["private", "address", "prefix"],
        "provenance": "RFC 1918, Section 3 address block recommendations",
    },
    {
        "id": "proto-006",
        "subdomain": "addressing",
        "question": "Why do the private address ranges of RFC 1918 require special routing considerations?",
        "document_url": RFC + "rfc1918.txt",
        "anchor": "non-connected",
        "expected_keywords": ["private", "enterprises", "internet"],
        "provenance": "RFC 1918, Section 2 rationale (global uniqueness vs private use)",
    },
    {
        "id": "proto-007",
        "subdomain": "addressing",
        "question": "Which IPv4 address blocks are reserved for use in documentation?",
        "document_url": RFC + "rfc5737.txt",
        "anchor": "192.0.2.",
        "expected_keywords": ["documentation", "examples", "address"],
        "provenance": "RFC 5737, Section 3 (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24)",
    },
    # -- arp ------------------------------------------------------------------
    {
        "id": "proto-008",
        "subdomain": "link-layer",
        "question": "What problem does ARP solve on a local network?",
        "document_url": RFC + "rfc826.txt",
        "anchor": "Converting Network Protocol Addresses",
        "expected_keywords": ["address", "resolution", "ethernet"],
        "provenance": "RFC 826, introduction (mapping protocol addresses to hardware addresses)",
    },
    # -- ipv6 -----------------------------------------------------------------
    {
        "id": "proto-009",
        "subdomain": "ipv6",
        "question": "What replaces the IPv4 Time to Live field in the IPv6 base header?",
        "document_url": RFC + "rfc8200.txt",
        "anchor": "renamed \"Hop Limit\" in IPv6",
        "expected_keywords": ["hop limit", "header", "version"],
        "provenance": "RFC 8200, Section 3 base header description",
    },
    # -- transport ------------------------------------------------------------
    {
        "id": "proto-010",
        "subdomain": "transport",
        "question": "Which three messages establish a TCP connection (the three-way handshake)?",
        "document_url": RFC + "rfc9293.txt",
        "anchor": "The \"three-way handshake\" is the procedure used to establish a",
        "expected_keywords": ["three-way", "handshake", "connection"],
        "provenance": "RFC 9293, Section 3.4 connection establishment (3W handshake)",
    },
    {
        "id": "proto-011",
        "subdomain": "transport",
        "question": "How does TCP terminate a connection cleanly?",
        "document_url": RFC + "rfc9293.txt",
        "anchor": "gracefully",
        "expected_keywords": ["close", "connection", "buffers"],
        "provenance": "RFC 9293, Section 3.5 connection termination",
    },
    {
        "id": "proto-012",
        "subdomain": "transport",
        "question": "What does the TCP retransmission mechanism protect against?",
        "document_url": RFC + "rfc9293.txt",
        "anchor": "detecting packet losses",
        "expected_keywords": ["retransmission", "loss", "checksums"],
        "provenance": "RFC 9293, reliability overview (acknowledgment and retransmission)",
    },
    # -- dns ------------------------------------------------------------------
    {
        "id": "proto-013",
        "subdomain": "dns",
        "question": "What is the role of a recursive name server in DNS resolution?",
        "document_url": RFC + "rfc1034.txt",
        "anchor": "is operating in recursive mode or not:",
        "expected_keywords": ["recursive", "queries", "answer"],
        "provenance": "RFC 1034, Section 2.3 resolvers vs servers",
    },
    {
        "id": "proto-014",
        "subdomain": "dns",
        "question": "What is a zone of authority in the DNS naming model?",
        "document_url": RFC + "rfc1034.txt",
        "anchor": "divided up into sections",
        "expected_keywords": ["zones", "database", "name servers"],
        "provenance": "RFC 1034, Section 4.3 zones",
    },
    {
        "id": "proto-015",
        "subdomain": "dns",
        "question": "What resource record types does a DNS query message carry?",
        "document_url": RFC + "rfc1035.txt",
        "anchor": "| QTYPE |",
        "expected_keywords": ["qtype", "qclass", "qname"],
        "provenance": "RFC 1035, Section 4.1.2 query message format",
    },
    {
        "id": "proto-016",
        "subdomain": "dns",
        "question": "How is a DNS name encoded in a message (labels and compression)?",
        "document_url": RFC + "rfc1035.txt",
        "anchor": "In order to reduce the size of messages, the domain system utilizes a",
        "expected_keywords": ["compression", "labels", "domain"],
        "provenance": "RFC 1035, Section 3.1 domain name representation and compression",
    },
    # -- dhcp -----------------------------------------------------------------
    {
        "id": "proto-017",
        "subdomain": "dhcp",
        "question": "Which message sequence does a DHCP client use to obtain an IP address?",
        "document_url": RFC + "rfc2131.txt",
        "anchor": "client DHCPDISCOVER client DHCPREQUEST client DHCPREQUEST",
        "expected_keywords": ["dhcpdiscover", "dhcprequest", "client"],
        "provenance": "RFC 2131, Section 3.1 client-server interaction (DORA)",
    },
    {
        "id": "proto-018",
        "subdomain": "dhcp",
        "question": "What does a DHCP server guarantee when it sends an ACK to a client request?",
        "document_url": RFC + "rfc2131.txt",
        "anchor": "respond with a DHCPACK",
        "expected_keywords": ["configuration", "parameters", "dhcpack"],
        "provenance": "RFC 2131, Section 3.1 DHCPACK semantics",
    },
    # -- routing --------------------------------------------------------------
    {
        "id": "proto-019",
        "subdomain": "routing",
        "question": "What are the states of the BGP finite state machine?",
        "document_url": RFC + "rfc4271.txt",
        "anchor": "refuses all incoming BGP",
        "expected_keywords": ["idle", "connect", "state"],
        "provenance": "RFC 4271, Section 8.1 BGP FSM overview",
    },
    {
        "id": "proto-020",
        "subdomain": "routing",
        "question": "What is the role of an OPEN message when a BGP session starts?",
        "document_url": RFC + "rfc4271.txt",
        "anchor": "In addition to the fixed-size BGP header, the OPEN message contains",
        "expected_keywords": ["open", "autonomous", "hold time"],
        "provenance": "RFC 4271, Section 4.2 OPEN message",
    },
    {
        "id": "proto-021",
        "subdomain": "routing",
        "question": "How does OSPF describe a network's topology?",
        "document_url": RFC + "rfc2328.txt",
        "anchor": "Each LSA describes a piece of the OSPF routing domain. Every router",
        "expected_keywords": ["lsa", "router", "link-state"],
        "provenance": "RFC 2328, Section 2.1 link-state advertisements",
    },
    # -- security -------------------------------------------------------------
    {
        "id": "proto-022",
        "subdomain": "security",
        "question": "Which messages are exchanged in the TLS 1.3 handshake?",
        "document_url": RFC + "rfc8446.txt",
        "anchor": "remaining bytes in message",
        "expected_keywords": ["clienthello", "serverhello", "handshake"],
        "provenance": "RFC 8446, Section 4 handshake overview",
    },
    {
        "id": "proto-023",
        "subdomain": "security",
        "question": "How does TLS 1.3 protect record integrity after the handshake?",
        "document_url": RFC + "rfc8446.txt",
        "anchor": "record protection functions",
        "expected_keywords": ["record", "protection", "plaintext"],
        "provenance": "RFC 8446, Section 5.2 record protection",
    },
    # -- http -----------------------------------------------------------------
    {
        "id": "proto-024",
        "subdomain": "http",
        "question": "What does the HTTP 405 status code indicate?",
        "document_url": RFC + "rfc9110.txt",
        "anchor": "Method Not Allowed) status",
        "expected_keywords": ["method", "allowed", "resource"],
        "provenance": "RFC 9110, Section 15.5.6 405 Method Not Allowed",
    },
    {
        "id": "proto-025",
        "subdomain": "http",
        "question": "What is the semantic difference between HTTP GET and HEAD?",
        "document_url": RFC + "rfc9110.txt",
        "anchor": "identical to GET",
        "expected_keywords": ["head", "get", "content"],
        "provenance": "RFC 9110, Section 9.3.2/9.3.3 GET and HEAD methods",
    },
    {
        "id": "proto-026",
        "subdomain": "http",
        "question": "What does the HTTP 204 status code mean?",
        "document_url": RFC + "rfc9110.txt",
        "anchor": "successfully fulfilled",
        "expected_keywords": ["content", "request", "204"],
        "provenance": "RFC 9110, Section 15.3.5 204 No Content",
    },
    # -- unsupported by construction (no expected ids; abstention tests) ------
    {
        "id": "proto-x01",
        "subdomain": "out-of-scope",
        "question": "Which RFC specifies the MQTT protocol?",
        "unsupported": True,
        "provenance": "unsupported: MQTT is an OASIS specification and is not part of this corpus",
    },
    {
        "id": "proto-x02",
        "subdomain": "out-of-scope",
        "question": "How do I configure a Cisco IOS interface with an access control list?",
        "unsupported": True,
        "provenance": "unsupported: vendor configuration is out of scope; no CLI content is ingested",
    },
    {
        "id": "proto-x03",
        "subdomain": "out-of-scope",
        "question": "What is CAN bus bit timing and how is it configured?",
        "unsupported": True,
        "provenance": "unsupported: CAN bus belongs to the automobile domain, not this corpus",
    },
    {
        "id": "proto-x04",
        "subdomain": "out-of-scope",
        "question": "Which RFC defines HTTP/2?",
        "unsupported": True,
        "provenance": "unsupported: RFC 7540/9113 (HTTP/2) is deliberately not ingested",
    },
    {
        "id": "proto-x05",
        "subdomain": "out-of-scope",
        "question": "What is the default OSPF hello interval on a Cisco router?",
        "unsupported": True,
        "provenance": "unsupported: vendor default, not specified by RFC 2328 in this corpus",
    },
    {
        "id": "proto-x06",
        "subdomain": "out-of-scope",
        "question": "Which RFC defines OAuth 2.0?",
        "unsupported": True,
        "provenance": "unsupported: RFC 6749 (OAuth 2.0) is not ingested",
    },
    {
        "id": "proto-x07",
        "subdomain": "out-of-scope",
        "question": "How do BGP routers authenticate with MD5 passwords in practice?",
        "unsupported": True,
        "provenance": "unsupported: operational password configuration is out of scope",
    },
]


# ---------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_v19_second_domain.py)
# ---------------------------------------------------------------------------


def find_grounding(anchor: str, document_url: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunks of `document_url` whose text contains `anchor` (case-insensitive)."""
    needle = (anchor or "").strip().lower()
    if not needle:
        return []
    return [
        c
        for c in chunks
        if c.get("document_url") == document_url and needle in (c.get("text") or "").lower()
    ]


def benchmark_from_spec(
    spec: list[dict[str, Any]],
    grounding: dict[str, list[dict[str, Any]]],
    *,
    kb_id: str,
    kb_name: str,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the benchmark artifact from the spec and its grounding matches."""
    authorship = dict(AUTHORSHIP)
    authorship["drafted_at"] = generated_at
    questions: list[dict[str, Any]] = []
    for item in spec:
        if item.get("unsupported"):
            questions.append(
                {
                    "id": item["id"],
                    "subdomain": item.get("subdomain", ""),
                    "question": item["question"],
                    "expected_chunk_ids": [],
                    "expected_document_ids": [],
                    "expected_keywords": [],
                    # Marker for the self-check and human readers: this question
                    # is an abstention probe, not a label we failed to ground.
                    # The benchmark schema ignores extra fields, so the runner
                    # still classifies it via `unlabelled_question_ids`.
                    "unsupported": True,
                    "provenance": item.get("provenance", ""),
                }
            )
            continue
        matches = grounding.get(item["id"], [])
        questions.append(
            {
                "id": item["id"],
                "subdomain": item.get("subdomain", ""),
                "question": item["question"],
                "expected_chunk_ids": [m["chunk_id"] for m in matches],
                "expected_document_ids": [matches[0]["document_id"]] if matches else [],
                "expected_keywords": list(item.get("expected_keywords", [])),
                "provenance": item.get("provenance", ""),
            }
        )
    return {
        "benchmark": "internet-protocols-draft-v1",
        "version": 1,
        "created_at": generated_at,
        "kb_id": kb_id,
        "kb_name": kb_name,
        "authorship": authorship,
        "relevance_model": "binary",
        "questions": questions,
    }


def grounding_problems(
    artifact: dict[str, Any], chunks: list[dict[str, Any]]
) -> list[str]:
    """Self-check: every label must be verifiable against the live corpus."""
    problems: list[str] = []
    by_id = {c["chunk_id"]: c for c in chunks}
    seen: set[str] = set()
    supported = 0
    for q in artifact.get("questions", []):
        qid = q.get("id", "")
        if qid in seen:
            problems.append(f"{qid}: duplicate question id")
        seen.add(qid)
        if q.get("unsupported"):
            continue
        supported += 1
        expected = q.get("expected_chunk_ids") or []
        if not expected:
            problems.append(f"{qid}: no expected chunk ids (supported question must be grounded)")
            continue
        for chunk_id in expected:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                problems.append(f"{qid}: expected chunk {chunk_id} is not in the corpus")
                continue
            text = (chunk.get("text") or "").lower()
            for keyword in q.get("expected_keywords", []):
                if keyword.lower() not in text:
                    problems.append(
                        f"{qid}: expected keyword {keyword!r} does not appear in chunk {chunk_id}"
                    )
    if supported == 0:
        problems.append("no supported question carries expected chunk ids: nothing is measurable")
    return problems


def _load_corpus(kb_id: str) -> list[dict[str, Any]]:
    """Every chunk of the KB with its document URL, for grounding."""
    from app.api.deps import get_repo

    repo = get_repo()
    docs = {d.id: d for d in repo.list_documents(kb_id)}
    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = repo.list_chunks(kb_id, limit=500, offset=offset)
        if not batch:
            break
        for chunk in batch:
            doc = docs.get(chunk.document_id)
            out.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_url": getattr(doc, "url", "") or "",
                    "text": chunk.text or "",
                }
            )
        offset += len(batch)
    return out


def _resolve_kb(args: argparse.Namespace) -> tuple[str, str]:
    if args.kb:
        kb = _repo().get_kb(args.kb)
        if kb is None:
            raise SystemExit(f"knowledge base {args.kb!r} not found")
        return kb.id, kb.name
    state_path = BACKEND_ROOT / "data" / "v19-validation-state.json"
    if state_path.is_file():
        kb_id = json.loads(state_path.read_text(encoding="utf-8")).get("kb_id") or ""
        if kb_id:
            kb = _repo().get_kb(kb_id)
            if kb is not None:
                return kb.id, kb.name
    raise SystemExit("pass --kb <kb_id> (or run scripts/v19_second_domain_validation.py first)")


def _repo():
    from app.api.deps import get_repo

    return get_repo()


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_draft(args: argparse.Namespace) -> int:
    kb_id, kb_name = _resolve_kb(args)
    chunks = _load_corpus(kb_id)
    print(f"kb {kb_id} ({kb_name}): {len(chunks)} chunk(s) available for grounding\n")
    bad = 0
    for item in QUESTIONS:
        if item.get("unsupported"):
            print(f"{item['id']:>10}  UNSUPPORTED  {item['question']}")
            continue
        matches = find_grounding(item["anchor"], item["document_url"], chunks)
        status = "OK" if len(matches) == 1 else ("AMBIGUOUS" if len(matches) else "NOT FOUND")
        if status != "OK":
            bad += 1
        print(f"{item['id']:>10}  {status:<10} anchor={item['anchor']!r} matches={len(matches)}")
        for match in matches[:3]:
            head = " ".join((match["text"]).split())[:110]
            print(f"{'':>14}{match['chunk_id']}  {head}")
    print(f"\n{bad} question(s) need anchor/document adjustments before finalize")
    return 1 if bad else 0


def cmd_finalize(args: argparse.Namespace) -> int:
    kb_id, kb_name = _resolve_kb(args)
    chunks = _load_corpus(kb_id)
    grounding = {
        item["id"]: find_grounding(item.get("anchor", ""), item.get("document_url", ""), chunks)
        for item in QUESTIONS
    }
    # Unsupported questions carry no anchor by design; only supported ones
    # must be grounded before the artifact can be written.
    ungrounded = [
        item["id"]
        for item in QUESTIONS
        if not item.get("unsupported") and not grounding.get(item["id"])
    ]
    if ungrounded:
        print(f"refusing to write: no grounding chunk for {ungrounded}")
        print("run the `draft` subcommand and adjust those anchors/documents")
        return 1
    artifact = benchmark_from_spec(
        QUESTIONS,
        grounding,
        kb_id=kb_id,
        kb_name=kb_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    problems = grounding_problems(artifact, chunks)
    if problems:
        print("refusing to write: grounding self-check failed")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    supported = sum(1 for q in artifact["questions"] if q["expected_chunk_ids"])
    print(
        f"wrote {BENCHMARK_PATH}\n"
        f"  {len(artifact['questions'])} questions ({supported} supported + grounded, "
        f"{len(artifact['questions']) - supported} unsupported by construction)\n"
        "  authorship: tier=draft, human_review=pending — exploratory use only"
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    kb_id, kb_name = _resolve_kb(args)
    if not BENCHMARK_PATH.is_file():
        print(f"no benchmark at {BENCHMARK_PATH}")
        return 1
    artifact = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    problems: list[str] = []
    if artifact.get("kb_id") != kb_id:
        problems.append(
            f"benchmark is bound to kb {artifact.get('kb_id')!r}, not {kb_id!r}"
        )
    authorship = artifact.get("authorship") or {}
    if authorship.get("tier") != "draft" or authorship.get("human_review") != "pending":
        problems.append(
            "authorship no longer says draft/pending — this artifact must not be "
            "presented as reviewed"
        )
    chunks = _load_corpus(kb_id)
    problems.extend(grounding_problems(artifact, chunks))

    from app.services.answer_eval.retrieval_diagnostic_runner import (
        file_sha256,
        load_retrieval_benchmark,
    )

    try:
        bench = load_retrieval_benchmark(BENCHMARK_PATH)
        fingerprint = bench.fingerprint()
    except Exception as exc:  # noqa: BLE001 - report, don't hide
        problems.append(f"the runner cannot load the artifact: {exc}")
        fingerprint = "unavailable"
    print(f"benchmark: {artifact.get('benchmark')} v{artifact.get('version')}")
    print(f"bound to : {artifact.get('kb_id')} ({kb_name})")
    print(f"sha256   : {file_sha256(BENCHMARK_PATH)}")
    print(f"fingerprint: {fingerprint}")
    print(f"questions: {len(artifact.get('questions', []))}")
    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nevery label is grounded in the live corpus and the artifact is a draft.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("draft", "finalize", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--kb", default="", help="knowledge base id (default: from the V19 state file)")
    args = parser.parse_args(argv)
    if args.command == "draft":
        return cmd_draft(args)
    if args.command == "finalize":
        return cmd_finalize(args)
    return cmd_verify(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
