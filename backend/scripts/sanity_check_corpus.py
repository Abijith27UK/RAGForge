"""SAFE live integration check for the V5 corpus workflow (Phase 12).

Runs the whole corpus lifecycle against a LIVE backend with REAL Qdrant and REAL
MiniLM embeddings, using:

  * a temporary knowledge base with a unique id,
  * its own isolated Qdrant collection (``kb_<temporary id>``),
  * temporary documents built in memory,
  * full cleanup of the collection, the SQLite rows and the uploaded files on exit.

It NEVER touches a pre-existing knowledge base or experiment collection. Every
destructive step runs against the temporary KB only.

    cd backend && .venv/Scripts/python.exe scripts/sanity_check_corpus.py --base http://localhost:8011
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(label: str, condition: bool, evidence: str = "") -> None:
    if condition:
        PASSED.append(label)
        print(f"PASS   {label}\n        {evidence}")
    else:
        FAILED.append((label, evidence))
        print(f"FAIL   {label}\n        {evidence}")


def docx_bytes() -> bytes:
    """A real .docx with heading sections (not a renamed ZIP)."""
    sys.path.insert(0, str(BACKEND_DIR / "tests"))
    from tests.test_user_ingestion import make_docx

    body = " ".join(
        f"Paragraph {n} develops the theory of resistance and its practical use."
        for n in range(1, 12)
    )
    return make_docx([("heading", "Resistance"), ("body", body),
                      ("heading", "Stability"), ("body", body)])


def pptx_bytes() -> bytes:
    sys.path.insert(0, str(BACKEND_DIR / "tests"))
    from tests.test_user_ingestion import make_pptx

    body = " ".join(
        f"Bullet {n} explains the free surface effect in detail for students."
        for n in range(1, 12)
    )
    return make_pptx([("Free Surface Effect", [body]), ("Ship Stability", [body])])


def lecture_text(n: int) -> bytes:
    return (
        f"# Lecture {n}\n\n"
        "Buoyancy and displacement relate through Archimedes' principle for a "
        "vessel of given volume and mean draft. The waterplane area enters the "
        "displacement equation directly. The block coefficient links volume and "
        f"draft for lecture {n}. " * 4
    ).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8011")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    import httpx

    kb_id: str | None = None
    with httpx.Client(base_url=base, timeout=600.0) as c:
        # -- 0. the backend must actually be up -----------------------------
        r = c.get("/api/system/health")
        check("Backend reachable", r.status_code == 200, f"GET /api/system/health -> {r.status_code}")
        if r.status_code != 200:
            return 1
        # Dependency reachability lives on /status, not /health.
        st = c.get("/api/system/status")
        qdrant_ok = st.json().get("qdrant", {}).get("reachable") if st.status_code == 200 else None
        check("Qdrant reachable (live test, not skipped)", bool(qdrant_ok),
              f"qdrant.reachable={qdrant_ok} at {st.json().get('qdrant', {}).get('url', '?')}")

        # -- 1. create an ISOLATED temporary knowledge base -----------------
        r = c.post("/api/knowledge-bases", json={
            "name": "SANITY-CORPUS-TEMP (safe integration check)",
            "domain": "Naval Architecture",
            "purpose": "Temporary corpus engineering verification",
            "target_audience": "engineers",
            "source_mode": "user_provided",
        })
        check("Create isolated temporary KB", r.status_code == 201, f"HTTP {r.status_code}")
        if r.status_code != 201:
            print(r.text[:400])
            return 1
        kb = r.json()
        kb_id = kb["id"]

        # -- 2. bulk ingestion (mixed formats + one deliberate failure) -----
        def part(name: str, data: bytes, ctype: str):
            """httpx multipart part: (filename, file-like, content-type)."""
            return (name, io.BytesIO(data), ctype)

        files = [
            part("lecture_01.txt", lecture_text(1), "text/plain"),
            part("lecture_02.md", lecture_text(2), "text/markdown"),
            part("lecture_03.txt", lecture_text(3), "text/plain"),
            part("resistance.docx", docx_bytes(),
                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            part("stability.pptx", pptx_bytes(),
                 "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
            part("broken.pptx", b"this is not a zip container", "application/octet-stream"),
            part("lecture_04.txt", lecture_text(4), "text/plain"),
        ]
        r = c.post(f"/api/knowledge-bases/{kb_id}/ingestion-batches",
                   files=[("files", f) for f in files],
                   data={"index": "true"})
        check("Bulk batch ingestion", r.status_code == 201, f"HTTP {r.status_code}")
        if r.status_code != 201:
            print(r.text[:600])
            return 1
        detail = r.json()
        b = detail["batch"]
        check("Batch accounted for every file",
              b["total_items"] == len(files) and
              b["completed_items"] + b["failed_items"] == len(files),
              f"total={b['total_items']} completed={b['completed_items']} "
              f"failed={b['failed_items']} status={b['status']}")
        check("One bad file does NOT fail the batch",
              b["completed_items"] == 6 and b["failed_items"] == 1 and b["status"] == "partial",
              f"partial batch: {b['completed_items']} ok / {b['failed_items']} failed / "
              f"status={b['status']}")
        bad = [i for i in detail["items"] if i["status"] == "failed"]
        check("Failure carries an actionable reason",
              bool(bad) and bool(bad[0]["error_code"]) and bool(bad[0]["error_message"]),
              f"{bad[0]['error_code'] if bad else '-'}: {bad[0]['error_message'][:70] if bad else ''}")

        # -- 3. provenance --------------------------------------------------
        r = c.get(f"/api/knowledge-bases/{kb_id}/corpus-manifest")
        manifest = r.json()
        summary = manifest["summary"]
        check("Manifest counts the corpus",
              summary["total_documents"] == 6 and summary["total_chunks"] > 0,
              f"documents={summary['total_documents']} chunks={summary['total_chunks']} "
              f"vectors={summary['total_vectors']} confirmed={summary['vector_count_confirmed']}")
        check("Vector count is CONFIRMED against live Qdrant",
              summary["vector_count_confirmed"] is True,
              f"total_vectors={summary['total_vectors']}")
        by_name = {e["file_name"]: e for e in manifest["entries"]}
        pptx_entry = by_name.get("stability.pptx", {})
        docx_entry = by_name.get("resistance.docx", {})
        check("PPTX slide provenance survives the corpus pipeline",
              bool(pptx_entry.get("slide_count")),
              f"slides={pptx_entry.get('slide_count')} parser={pptx_entry.get('parser')}")
        check("DOCX section provenance survives the corpus pipeline",
              bool(docx_entry.get("section_count")),
              f"sections={docx_entry.get('section_count')} parser={docx_entry.get('parser')}")

        # -- 4. duplicate detection ----------------------------------------
        r = c.post(f"/api/knowledge-bases/{kb_id}/ingestion-batches",
                   files=[("files", part("lecture_01.txt", lecture_text(1), "text/plain"))],
                   data={"index": "true"})
        dup = r.json()
        check("Duplicate content detected, existing document kept",
              dup["batch"]["duplicate_items"] == 1
              and dup["batch"]["completed_items"] == 0,
              f"duplicates={dup['batch']['duplicate_items']} "
              f"completed={dup['batch']['completed_items']}")

        # -- 5. resume is idempotent ---------------------------------------
        docs_before = c.get(f"/api/knowledge-bases/{kb_id}/overview").json()["documents"]
        r = c.post(f"/api/knowledge-bases/{kb_id}/ingestion-batches/{b['id']}/resume",
                   json={"include_completed": False, "retry_failed": True, "index": True})
        resumed = r.json()
        docs_after = c.get(f"/api/knowledge-bases/{kb_id}/overview").json()["documents"]
        check("Resume is idempotent (no duplicate documents)",
              r.status_code == 200 and docs_after == docs_before,
              f"documents {docs_before} -> {docs_after}; "
              f"resumable={resumed['resumable_items']}")

        # -- 6. integrity scan ----------------------------------------------
        r = c.get(f"/api/knowledge-bases/{kb_id}/corpus-integrity")
        report = r.json()
        check("Integrity scan ran against live vectors",
              r.status_code == 200 and report["vectors_checked"] != -1,
              f"status={report['overall_status']} docs={report['documents_checked']} "
              f"chunks={report['chunks_checked']} vectors={report['vectors_checked']}")
        check("Integrity scan is clean on a fresh corpus",
              not report["orphan_vectors"] and not report["missing_vectors"]
              and not report["stale_vectors"],
              f"orphans={len(report['orphan_vectors'])} "
              f"missing={len(report['missing_vectors'])} "
              f"stale={len(report['stale_vectors'])}")

        # -- 7. repair is refused without confirmation -----------------------
        r = c.post(f"/api/knowledge-bases/{kb_id}/corpus-repair", json={"action": "rebuild_kb"})
        check("Destructive repair refused without confirm_action",
              r.status_code == 400 and "confirm_action" in r.json().get("detail", ""),
              f"HTTP {r.status_code}: {r.json().get('detail', '')[:80]}")

        # -- 8. fingerprint determinism -------------------------------------
        f1 = c.get(f"/api/knowledge-bases/{kb_id}/corpus-fingerprint").json()
        f2 = c.get(f"/api/knowledge-bases/{kb_id}/corpus-fingerprint").json()
        check("Corpus fingerprint is deterministic",
              f1["fingerprint"] == f2["fingerprint"],
              f"{f1['fingerprint'][:16]}… ({f1['document_count']} docs)")
        v1 = c.post(f"/api/knowledge-bases/{kb_id}/corpus-versions", json={"note": "sanity"})
        v2 = c.post(f"/api/knowledge-bases/{kb_id}/corpus-versions", json={"note": "sanity again"})
        check("Snapshotting an unchanged corpus reuses one version",
              v1.json()["id"] == v2.json()["id"],
              f"{v1.json()['version']} / fingerprint {v1.json()['fingerprint'][:12]}…")

        # -- 9. ground truth honesty ----------------------------------------
        overview = c.get(f"/api/knowledge-bases/{kb_id}/overview").json()
        check("New domain reports ground truth NOT_AVAILABLE",
              overview["ground_truth_status"] == "NOT_AVAILABLE"
              and overview["last_evaluation"] is None,
              f"status={overview['ground_truth_status']} "
              f"questions={overview['ground_truth_question_count']}; no metrics reported")

        # -- 10. retrieval still traces back to the source ------------------
        r = c.post(f"/api/knowledge-bases/{kb_id}/retrieve", json={
            "query": "free surface effect on ship stability", "top_k": 3,
        })
        hits = r.json().get("results", [])
        check("Retrieval returns provenance-bearing results",
              r.status_code == 200 and len(hits) > 0
              and all(h.get("document_id") for h in hits),
              f"{len(hits)} results; top doc={hits[0].get('document_file') or hits[0].get('document_id', '')[:12]} "
              f"page={hits[0].get('page')} slide={hits[0].get('slide')} "
              f"score={round(hits[0].get('score', 0), 4)}" if hits else "no results")

    # -- cleanup: this KB and its collection only ---------------------------
    if kb_id:
        with httpx.Client(base_url=base, timeout=120.0) as c:
            c.delete(f"/api/knowledge-bases/{kb_id}")
        print(f"\nCleaned up temporary KB {kb_id} (rows + collection + files)")

    print("\n" + "=" * 72)
    if FAILED:
        print(f"{len(PASSED)} passed, {len(FAILED)} FAILED")
        for label, ev in FAILED:
            print(f"  FAILED: {label} — {ev}")
        return 1
    print(f"{len(PASSED)}/{len(PASSED)} corpus checks passed — ALL PASS")
    print(f"Isolated KB: {kb_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())