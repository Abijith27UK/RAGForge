"""Throwaway end-to-end sanity check for the V4 user-knowledge workflow.

Exercises all 15 product capabilities against a LIVE backend, with real
SentenceTransformers embeddings and a real Qdrant instance. Prints one line per
capability with PASS/FAIL and the evidence it observed.

This is a verification harness, not a test suite (the pytest suite covers the
hermetic cases). Run it against a live server:

    python scripts/sanity_check_user_kb.py --base http://localhost:8011
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import httpx

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "tests"))

RESULTS: list[tuple[str, bool, str]] = []


def check(num: int, name: str, ok: bool, evidence: str) -> None:
    RESULTS.append((f"{num:>2}. {name}", ok, evidence))
    print(f"{'PASS' if ok else 'FAIL'}  {num:>2}. {name}\n        {evidence}")


def build_fixtures(tmp: Path) -> dict[str, tuple[str, bytes, str]]:
    """Real files built with the real libraries."""
    from test_user_ingestion import make_docx, make_pdf_like_bytes, make_pptx

    gm = "The metacentric height GM equals KB plus BM minus KG for small angles of heel."
    return {
        "stability.pptx": (
            "stability.pptx", make_pptx([
                ("Ship Stability", [gm + " A vessel is stable while the metacentre stays above KG."]),
                ("Free Surface Effect", ["Free surface effect reduces the effective metacentric height "
                                         "whenever a tank is only partly filled with a lighter liquid."]),
            ]), "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        "notes.docx": (
            "notes.docx", make_docx([
                ("heading", "Trim and List"),
                ("body", "Trim is a longitudinal moment produced by weight distribution along the length."),
                ("heading", "Resistance"),
                ("body", "Wave-making resistance becomes significant at higher Froude numbers."),
            ]), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "paper.pdf": ("paper.pdf", make_pdf_like_bytes(), "application/pdf"),
        "class_notes.md": (
            "class_notes.md",
            b"# Class Notes\n\nBuoyancy equals the weight of the fluid displaced by the hull form.\n",
            "text/markdown"),
        "plain.txt": (
            "plain.txt",
            b"Hydrostatics deals with the forces acting on a partially submerged floating body.\n",
            "text/plain"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8011")
    args = ap.parse_args()
    tmp = Path(__file__).resolve().parent.parent / "data" / "_sanity"
    tmp.mkdir(parents=True, exist_ok=True)
    c = httpx.Client(base_url=args.base, timeout=900.0)

    try:
        # 1. Create a KB --------------------------------------------------
        r = c.post("/api/knowledge-bases", json={
            "name": "SANITY Naval Architecture", "domain": "Naval Architecture",
            "purpose": "Verify the user-provided knowledge workflow end to end",
            "target_audience": "Naval Architecture students",
            "source_mode": "user_provided"})
        kb = r.json()
        kb_id = kb["id"]
        check(1, "Create a knowledge base", r.status_code == 201,
              f"HTTP {r.status_code} kb_id={kb_id}")

        # 2. USER_PROVIDED mode ------------------------------------------
        check(2, "Source mode = USER_PROVIDED", kb.get("source_mode") == "user_provided",
              f"source_mode={kb.get('source_mode')}")

        # 3 + 4 + 5. Upload, validate, parse -------------------------------
        fixtures = build_fixtures(tmp)
        files = [("files", (n, b, ct)) for n, (fn, b, ct) in fixtures.items()]
        # add an invalid file to prove per-file validation
        files.append(("files", ("broken.pptx", b"not a real powerpoint at all", "application/octet-stream")))
        t0 = time.perf_counter()
        r = c.post(f"/api/knowledge-bases/{kb_id}/documents/upload", files=files, data={"index": "true"})
        up = r.json()
        elapsed = time.perf_counter() - t0
        by_name = {f["file_name"]: f for f in up["files"]}
        check(3, "Upload multiple PDF/PPTX/DOCX/MD/TXT",
              up["uploaded"] == 5,
              f"uploaded={up['uploaded']} (stability.pptx, notes.docx, paper.pdf, "
              f"class_notes.md, plain.txt) in {elapsed:.1f}s")
        check(4, "File validation (per-file, no batch abort)",
              up["rejected"] == 1 and by_name["broken.pptx"]["status"] == "rejected",
              f"rejected={up['rejected']}: {by_name['broken.pptx']['message'][:70]}")
        parsed_ok = all(
            by_name[n]["document"]["text_length"] > 0
            for n in ("stability.pptx", "notes.docx", "paper.pdf", "class_notes.md", "plain.txt")
        )
        check(5, "Parsed into normalized text", parsed_ok,
              "text_length>0 for all 5: " +
              ", ".join(f"{n.split('.')[0]}={by_name[n]['document']['text_length']}ch"
                        for n in by_name if by_name[n]["document"]))

        # 6. Provenance: page / slide / section -----------------------------
        detail = c.get(f"/api/knowledge-bases/{kb_id}/documents/"
                       f"{by_name['stability.pptx']['document']['id']}").json()
        ppt_chunks = c.get(f"/api/knowledge-bases/{kb_id}/documents/"
                           f"{by_name['stability.pptx']['document']['id']}/chunks").json()
        docx_id = by_name["notes.docx"]["document"]["id"]
        docx_chunks = c.get(f"/api/knowledge-bases/{kb_id}/documents/{docx_id}/chunks").json()
        pdf_id = by_name["paper.pdf"]["document"]["id"]
        pdf_chunks = c.get(f"/api/knowledge-bases/{kb_id}/documents/{pdf_id}/chunks").json()
        slides = sorted({ch["slide"] for ch in ppt_chunks if ch["slide"]})
        titles = [ch["slide_title"] for ch in ppt_chunks if ch["slide_title"]]
        pages = sorted({ch["page"] for ch in pdf_chunks if ch["page"]})
        paths = [ch["section_path"] for ch in docx_chunks if ch["section_path"]]
        check(6, "page / slide / section provenance",
              slides == [1, 2] and pages == [1, 2] and any("Trim and List" in p for p in paths),
              f"pptx slides={slides} titles={sorted(set(titles))} | "
              f"pdf pages={pages} | docx section_paths={paths[:3]}")

        # 7 + 8 + 9. Chunked, embedded, indexed in Qdrant -----------------
        idx = up.get("indexing") or {}
        overview = c.get(f"/api/knowledge-bases/{kb_id}/overview").json()
        qdrant_ok = overview["vectors"] is not None and overview["vectors"] > 0
        check(7, "Chunked", overview["chunks"] > 0,
              f"{overview['chunks']} chunks across {overview['documents']} documents")
        check(8, "Embedded (real sentence-transformers)",
              idx.get("vectors_indexed", 0) > 0,
              f"{idx.get('vectors_indexed')} vectors via "
              f"{overview['embedding_model']} ({idx.get('seconds')}s)")
        check(9, "Stored in Qdrant", qdrant_ok,
              f"Qdrant points_count={overview['vectors']} (store status: "
              f"{overview['vector_store_status']}), SQLite chunks={overview['chunks']}")

        # 10. Incremental add ---------------------------------------------
        before_chunks = overview["chunks"]
        before_points = overview["vectors"]
        extra = c.post(f"/api/knowledge-bases/{kb_id}/documents/upload",
                       files=[("files", ("extra_lesson.md",
                                         b"# Added Later\n\nCavitation begins when local pressure "
                                         b"falls below the vapour pressure of the water.\n",
                                         "text/markdown"))],
                       data={"index": "true"}).json()
        o2 = c.get(f"/api/knowledge-bases/{kb_id}/overview").json()
        incremental = extra["indexing"]
        check(10, "Incrementally add one document",
              incremental["documents_indexed"] == 1
              and o2["documents"] == overview["documents"] + 1
              and o2["chunks"] > before_chunks,
              f"added doc #{o2['documents']}: only {incremental['documents_indexed']} document "
              f"indexed, +{o2['chunks'] - before_chunks} chunks, "
              f"vectors {before_points} -> {o2['vectors']}; version v{o2['version']}")

        # 11. Replace a document -------------------------------------------
        target = by_name["plain.txt"]["document"]["id"]
        r = c.post(f"/api/knowledge-bases/{kb_id}/documents/{target}/replace",
                   files=[("file", ("plain_v2.txt",
                                    b"Hydrostatics covers buoyancy, stability, trim and the "
                                    b"metacentric height of floating bodies.\n", "text/plain"))],
                   data={"index": "true"})
        rep = r.json()
        hist = c.get(f"/api/knowledge-bases/{kb_id}/documents/{target}").json()["document"]
        check(11, "Replace a document (new version)",
              r.status_code == 200 and rep["document"]["document_version"] == 2,
              f"v{rep['document']['document_version']} supersedes {rep['superseded'][:12]}…; "
              f"old kept as history with status={hist['status']}")

        # 12. Delete a document + its vectors ------------------------------
        del_id = by_name["class_notes.md"]["document"]["id"]
        o3 = c.get(f"/api/knowledge-bases/{kb_id}/overview").json()
        r = c.delete(f"/api/knowledge-bases/{kb_id}/documents/{del_id}")
        d = r.json()
        o4 = c.get(f"/api/knowledge-bases/{kb_id}/overview").json()
        check(12, "Delete a document and its vectors",
              r.status_code == 200 and o4["documents"] == o3["documents"] - 1
              and o4["vectors"] < o3["vectors"],
              f"vectors_removed={d['vectors_removed']}, store_error={d['vector_store_error']}; "
              f"Qdrant {o3['vectors']} -> {o4['vectors']}, docs {o3['documents']} -> {o4['documents']}")

        # 13. Inspect extracted text / chunks ------------------------------
        txt = c.get(f"/api/knowledge-bases/{kb_id}/documents/{ppt_chunks[0]['document_id']}/text").json()
        ins = c.get(f"/api/knowledge-bases/{kb_id}/documents/"
                    f"{by_name['stability.pptx']['document']['id']}").json()
        check(13, "Inspect extracted text and chunks",
              txt["returned"] > 0 and ins["source"]["user_provided"] is True
              and ins["chunk_count"] > 0,
              f"text={txt['returned']}/{txt['char_count']} chars via parser='{txt['parser']}'; "
              f"chunks={ins['chunk_count']}; integrity assessed_by="
              f"{ins['integrity']['assessed_by']}")

        # 14 + 15. Retrieve and see provenance ------------------------------
        q = "What is the effect of free surface on ship stability?"
        ret = c.post(f"/api/knowledge-bases/{kb_id}/retrieve",
                     json={"query": q, "top_k": 5}).json()
        top = ret["results"][0] if ret["results"] else None
        prov = top["provenance"] if top else {}
        check(14, "Retrieve chunks (Retrieval Lab)",
              len(ret["results"]) > 0,
              f"query={q!r} -> {len(ret['results'])} results, backend={ret['retrieval_backend']}, "
              f"embedding={ret['embedding_model']}")
        full_chain = all(prov.get(k) for k in
                         ("document_id", "document_title", "chunking_strategy"))
        has_unit = prov.get("slide") is not None or prov.get("page") is not None
        check(15, "Provenance visible on retrieved chunks",
              full_chain and has_unit and top["chunk_id"],
              f"top hit: doc={prov.get('document_title')} slide={prov.get('slide')} "
              f"({prov.get('slide_title')}) page={prov.get('page')} "
              f"section={prov.get('section_path')} score={top['score']} "
              f"chunk={top['chunk_id']} user_provided={prov.get('user_provided')}")

        print("\n" + "=" * 72)
        failed = [n for n, ok, _ in RESULTS if not ok]
        print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} capabilities verified"
              + (f" — FAILED: {failed}" if failed else " — ALL PASS"))
        print(f"KB under test: {kb_id}")
        c.delete(f"/api/knowledge-bases/{kb_id}")
        print(f"cleaned up KB {kb_id} (collection + rows + files)")
        return 1 if failed else 0
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(main())