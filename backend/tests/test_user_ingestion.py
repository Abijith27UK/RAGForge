"""V4 user-knowledge ingestion tests (Phases 2, 3, 4).

Covers: PPTX/DOCX/TXT/MD/HTML ingestion, slide/section/page provenance,
upload validation (name safety, size, magic bytes, corrupt OOXML), duplicate
detection, invalid documents, and the USER_PROVIDED integrity model.

All tests are hermetic: no network, no Qdrant, real parser libraries on real
in-memory OOXML files built with python-pptx / python-docx.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.models import Source, SourceType  # noqa: E402
from app.services.chunking.chunker import get_chunker, split_into_sections  # noqa: E402
from app.services.ingestion.ingestion import (  # noqa: E402
    IngestionError,
    ingest_uploaded_bytes,
)
from app.services.ingestion.parsers import (  # noqa: E402
    PARSER_REGISTRY,
    DocxParser,
    IngestionError as ParserIngestionError,
    LegacyPptParser,
    MarkdownParser,
    PdfParser,
    PptxParser,
    TextParser,
    get_parser,
    supported_upload_extensions,
)
from app.services.ingestion.upload import (  # noqa: E402
    sanitize_file_name,
    validate_upload,
)
from app.services.source_quality.user_scorer import (  # noqa: E402
    UserProvidedIntegrityScorer,
    assess_user_upload,
)
from app.utils.ids import new_id  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builders — real files, built in memory
# ---------------------------------------------------------------------------

def make_pptx(slides: list[tuple[str, list[str]]]) -> bytes:
    """slides: [(slide_title, [bullet lines]), ...] -> real .pptx bytes."""
    from pptx import Presentation

    prs = Presentation()
    title_only = prs.slide_layouts[5]  # has a real title placeholder
    for title, bullets in slides:
        slide = prs.slides.add_slide(title_only)
        slide.shapes.title.text = title
        body = slide.shapes.add_textbox(0, 1_200_000, 6_000_000, 3_000_000)
        tf = body.text_frame
        tf.text = bullets[0] if bullets else ""
        for extra in bullets[1:]:
            tf.add_paragraph().text = extra
    buffer = io.BytesIO()
    prs.save(buffer)
    return buffer.getvalue()


def make_docx(blocks: list[tuple[str, str]]) -> bytes:
    """blocks: [("heading", "Ship Stability"), ("body", "..."), ...] -> .docx bytes."""
    import docx

    document = docx.Document()
    for kind, text in blocks:
        if kind == "heading":
            document.add_heading(text, level=1)
        elif kind == "heading2":
            document.add_heading(text, level=2)
        elif kind == "body":
            document.add_paragraph(text)
        elif kind == "table":
            table = document.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "Condition"
            table.cell(0, 1).text = "Free surface effect"
            table.cell(1, 0).text = text
            table.cell(1, 1).text = "Increases metacentric radius loss"
        else:  # pragma: no cover
            raise AssertionError(f"unknown block kind {kind}")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_pdf_like_bytes() -> bytes:
    """A minimal but structurally valid PDF (2 pages, extractable text)."""
    # Hand-built PDF with two pages of visible text.
    objects = []
    content1 = (
        b"BT /F1 18 Tf 72 700 Td (Ship Stability Fundamentals) Tj ET\n"
        b"BT /F1 12 Tf 72 650 Td (A floating body is stable when the metacentre stays above KG.) Tj ET"
    )
    content2 = (
        b"BT /F1 18 Tf 72 700 Td (Free Surface Effect) Tj ET\n"
        b"BT /F1 12 Tf 72 650 Td (Free surface effect reduces stability by lowering effective GM.) Tj ET"
    )
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>"
    )
    objects.append(b"<< /Length " + str(len(content1)).encode() + b" >>\nstream\n" + content1 + b"\nendstream")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>"
    )
    objects.append(b"<< /Length " + str(len(content2)).encode() + b" >>\nstream\n" + content2 + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


def user_source(file_name: str, size: int = 1234) -> Source:
    return Source(
        id=new_id("src"),
        kb_id="kb_v4",
        url=f"upload://{file_name}",
        title=file_name,
        source_type=SourceType.PRESENTATION,
        user_provided=True,
        provenance="user_upload",
        file_name=file_name,
        file_size=size,
    )


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

def test_registry_resolves_every_advertised_extension():
    for ext in supported_upload_extensions():
        assert ext.startswith("."), ext
    assert get_parser(Path("deck.pptx")).name == "pptx"
    assert get_parser(Path("notes.docx")).name == "docx"
    assert get_parser(Path("book.pdf")).name == "pdf"
    assert get_parser(Path("a.md")).name == "markdown"
    assert get_parser(Path("a.txt")).name == "txt"
    assert get_parser(Path("a.html")).name == "html"


def test_unknown_extension_fails_loudly():
    with pytest.raises(ParserIngestionError, match="Unsupported file type"):
        get_parser(Path("data.xlsx"))


def test_parser_resolution_falls_back_to_content_type():
    parser = get_parser(Path("download"), content_type="application/pdf")
    assert isinstance(parser, PdfParser)


# ---------------------------------------------------------------------------
# Phase 3: PPTX ingestion
# ---------------------------------------------------------------------------

def test_pptx_ingestion_records_slide_count_and_slide_titles(tmp_path):
    data = make_pptx(
        [
            ("Ship Stability", ["Archimedes principle governs buoyancy of a floating body.", "GM = KB + BM - KG for small heel angles."]),
            ("Free Surface Effect", ["Free surface correction reduces the effective metacentric height."]),
            ("Trim and List", ["Trim is a longitudinal moment while list is a transverse moment."]),
        ]
    )
    source = user_source("NA5010_Lecture_08.pptx", len(data))
    doc = ingest_uploaded_bytes(
        kb_id="kb_v4", source=source, upload_dir=tmp_path,
        file_name="NA5010_Lecture_08.pptx", data=data,
    )
    assert doc.slide_count == 3
    assert doc.page_count is None, "a deck has slides, not pages — never fabricate pages"
    assert doc.parser == "pptx"
    assert doc.user_provided is True
    assert doc.status.value == "parsed"
    assert "Archimedes principle governs buoyancy" in Path(doc.file_path).read_text(encoding="utf-8")
    assert doc.parse_metadata["slide_titles"][1] == "Free Surface Effect"
    # Original bytes are preserved for rebuild.
    assert Path(doc.raw_file_path).read_bytes() == data


def test_pptx_chunks_carry_real_slide_numbers_and_titles(tmp_path):
    data = make_pptx(
        [
            ("Ship Stability", ["Buoyancy equals the weight of the fluid displaced by the hull."]),
            ("Free Surface Effect", ["Free surface effect reduces the effective metacentric height of a partly filled tank."]),
        ]
    )
    source = user_source("stability.pptx", len(data))
    doc = ingest_uploaded_bytes(
        kb_id="kb_v4", source=source, upload_dir=tmp_path,
        file_name="stability.pptx", data=data,
    )
    text = Path(doc.file_path).read_text(encoding="utf-8")
    chunks = get_chunker("section-aware").chunk(
        document=doc, text=text, source=source, target_size=400, overlap=50,
        domain="Naval Architecture",
        chunking_config={"target_size": 400, "overlap": 50},
    )
    assert chunks, "a slide deck must produce chunks"
    free_surface = next(c for c in chunks if "metacentric" in c.text)
    assert free_surface.slide == 2, "slide number must come from the file, not be invented"
    assert free_surface.page is None, "slides are not pages"
    assert free_surface.slide_title == "Free Surface Effect"
    assert free_surface.section_path and "Free Surface Effect" in free_surface.section_path
    assert free_surface.chunking_strategy == "section-aware"
    assert free_surface.chunking_config["target_size"] == 400
    assert free_surface.document_version == 1
    assert free_surface.user_provided is True
    # Markers are stripped from chunk text.
    assert "[SLIDE" not in free_surface.text


def test_pptx_notes_and_tables_are_captured(tmp_path):
    from pptx import Presentation
    from pptx.util import Emu

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title-only
    slide.shapes.title.text = "Hydrostatics"
    notes = slide.notes_slide
    notes.notes_text_frame.text = "Explain the freshwater vs seawater density difference."
    rows, cols = 2, 2
    table = slide.shapes.add_table(rows, cols, Emu(100000), Emu(2000000), Emu(3000000), Emu(1000000)).table
    table.cell(0, 0).text = "Fluid"
    table.cell(0, 1).text = "Density kg/m3"
    table.cell(1, 0).text = "Seawater"
    table.cell(1, 1).text = "1025"
    buffer = io.BytesIO()
    prs.save(buffer)

    parser = PptxParser()
    text, meta = parser.parse(Path("lecture.pptx"), buffer.getvalue())
    assert "freshwater vs seawater" in text
    assert "| Seawater | 1025 |" in text
    assert meta["slide_count"] == 1


def test_legacy_ppt_is_rejected_with_an_actionable_message(tmp_path):
    parser = LegacyPptParser()
    data = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 200
    with pytest.raises(IngestionError, match="Re-save the deck as .pptx"):
        parser.parse(Path("old.ppt"), data)


# ---------------------------------------------------------------------------
# Phase 3: DOCX ingestion
# ---------------------------------------------------------------------------

def test_docx_ingestion_preserves_heading_path_and_tables(tmp_path):
    data = make_docx(
        [
            ("heading", "Ship Stability"),
            ("body", "A ship is stable when the metacentre stays above the centre of gravity."),
            ("heading2", "Free Surface Effect"),
            ("body", "Loose liquids reduce the effective metacentric height."),
            ("table", "Full tank"),
            ("heading", "Trim and List"),
            ("body", "Trim is a longitudinal moment while list is a transverse moment."),
        ]
    )
    source = Source(
        id=new_id("src"), kb_id="kb_v4", url="upload://notes.docx", title="notes.docx",
        source_type=SourceType.DOCUMENT, user_provided=True, provenance="user_upload",
    )
    doc = ingest_uploaded_bytes(
        kb_id="kb_v4", source=source, upload_dir=tmp_path,
        file_name="notes.docx", data=data,
    )
    assert doc.parser == "docx"
    assert doc.section_count == 3
    body = Path(doc.file_path).read_text(encoding="utf-8")
    assert "# Ship Stability" in body, "headings must survive as structure"
    assert "## Free Surface Effect" in body
    assert "| Condition | Free surface effect |" in body, "tables must not be destroyed"

    chunks = get_chunker("section-aware").chunk(
        document=doc, text=body, source=source, target_size=500, overlap=60,
        domain="Naval Architecture",
    )
    fse = next(c for c in chunks if "effective metacentric height" in c.text)
    assert fse.section_path == "Ship Stability > Free Surface Effect"
    assert fse.section == "Free Surface Effect"


def test_docx_chunk_table_context_is_searchable(tmp_path):
    data = make_docx(
        [("heading", "Free Surface Tables"), ("table", "Partially full tank")]
    )
    source = Source(
        id=new_id("src"), kb_id="kb_v4", url="upload://t.docx", title="t.docx",
        source_type=SourceType.DOCUMENT, user_provided=True, provenance="user_upload",
    )
    doc = ingest_uploaded_bytes(
        kb_id="kb_v4", source=source, upload_dir=tmp_path, file_name="t.docx", data=data
    )
    body = Path(doc.file_path).read_text(encoding="utf-8")
    chunks = get_chunker("section-aware").chunk(
        document=doc, text=body, source=source, target_size=400, overlap=40
    )
    assert any("Partially full tank" in c.text for c in chunks)


# ---------------------------------------------------------------------------
# Phase 3: PDF / TXT / MD / HTML (regression + provenance)
# ---------------------------------------------------------------------------

def test_pdf_ingestion_records_page_count(tmp_path):
    data = make_pdf_like_bytes()
    source = Source(
        id=new_id("src"), kb_id="kb_v4", url="upload://s.pdf", title="s.pdf",
        source_type=SourceType.PDF, user_provided=True, provenance="user_upload",
    )
    doc = ingest_uploaded_bytes(
        kb_id="kb_v4", source=source, upload_dir=tmp_path, file_name="s.pdf", data=data
    )
    assert doc.page_count == 2
    assert doc.slide_count is None
    text = Path(doc.file_path).read_text(encoding="utf-8")
    assert "[PAGE 1]" in text and "[PAGE 2]" in text
    chunks = get_chunker("section-aware").chunk(
        document=doc, text=text, source=source, target_size=400, overlap=40
    )
    page2 = next(c for c in chunks if "Free surface effect" in c.text)
    assert page2.page == 2
    assert page2.slide is None
    page1 = next(c for c in chunks if "metacentre" in c.text)
    assert page1.page == 1


def test_txt_and_markdown_keep_headings(tmp_path):
    md = b"# Hydrostatics\n\nArchimedes principle governs buoyancy in fluids.\n\n## Pressure\n\nPressure increases linearly with depth.\n"
    source = user_source("notes.md", len(md))
    doc = ingest_uploaded_bytes(
        kb_id="kb_v4", source=source, upload_dir=tmp_path, file_name="notes.md", data=md
    )
    assert doc.parser == "markdown"
    sections = split_into_sections(Path(doc.file_path).read_text(encoding="utf-8"))
    assert any(s.path == "Hydrostatics > Pressure" for s in sections)

    txt = b"Trim and list are distinct hydrostatic quantities used in ship design.\n"
    src2 = user_source("plain.txt", len(txt))
    doc2 = ingest_uploaded_bytes(
        kb_id="kb_v4", source=src2, upload_dir=tmp_path, file_name="plain.txt", data=txt
    )
    assert doc2.parser == "txt"
    assert doc2.text_length > 0


def test_html_parser_still_works(tmp_path):
    html = (
        b"<html><head><title>Ship Design</title></head><body><h1>Hull</h1>"
        b"<p>The hull form strongly influences resistance and stability.</p>"
        b"<script>var x=1;</script></body></html>"
    )
    source = user_source("page.html", len(html))
    doc = ingest_uploaded_bytes(
        kb_id="kb_v4", source=source, upload_dir=tmp_path, file_name="page.html", data=html
    )
    text = Path(doc.file_path).read_text(encoding="utf-8")
    assert "var x" not in text
    assert "Hull" in text


# ---------------------------------------------------------------------------
# Phase 2: upload validation
# ---------------------------------------------------------------------------

def test_sanitize_file_name_strips_traversal_and_unsafe_chars():
    assert sanitize_file_name("../../etc/passwd") == "passwd"
    assert sanitize_file_name(r"C:\Users\me\lecture 1.pptx") == "lecture 1.pptx"
    assert sanitize_file_name("notes;rm -rf.md") == "notes_rm -rf.md"
    assert sanitize_file_name("") == "upload"
    long_name = sanitize_file_name("a" * 400 + ".pptx")
    assert len(long_name) <= 180 and long_name.endswith(".pptx")


def test_validate_upload_accepts_real_pptx_and_docx():
    pptx = make_pptx([("Slide One", ["Content line for the slide body."])])
    ok = validate_upload("deck.pptx", pptx)
    assert ok.ok, ok.error
    assert ok.detected_format == "ooxml"

    docx = make_docx([("heading", "Section"), ("body", "Body text that is long enough.")])
    assert validate_upload("notes.docx", docx).ok


def test_validate_upload_rejects_empty_oversized_and_mislabelled():
    assert not validate_upload("a.pdf", b"").ok
    too_big = validate_upload("a.txt", b"hello", max_bytes=2)
    assert not too_big.ok and "above the" in too_big.error

    # A .pptx extension on a DOCX container: the required part is missing.
    docx = make_docx([("heading", "Section"), ("body", "Body text that is long enough.")])
    mismatch = validate_upload("fake.pptx", docx)
    assert not mismatch.ok
    assert "ppt/presentation.xml" in mismatch.error

    # Random bytes claiming to be a PDF.
    not_pdf = validate_upload("paper.pdf", b"this is definitely not a pdf file")
    assert not not_pdf.ok and "%PDF" in not_pdf.error

    assert not validate_upload("archive.zip", b"PK\x03\x04zzzz").ok
    assert not validate_upload("noextension", b"hello").ok


def test_validate_upload_detects_corrupt_ooxml():
    good = make_pptx([("Slide One", ["Body content for this slide."])])
    corrupted = bytearray(good)
    # Truncate: the ZIP central directory is gone -> not a readable container.
    result = validate_upload("deck.pptx", bytes(corrupted[: len(good) // 2]))
    assert not result.ok
    assert "ZIP" in result.error or "ppt/presentation.xml" in result.error


def test_pptx_without_title_placeholders_reports_no_slide_title(tmp_path):
    """No title placeholder -> no title. Never invent "Slide N"."""
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    slide.shapes.add_textbox(0, 0, 4_000_000, 2_000_000).text_frame.text = (
        "Body content on a slide that has no title placeholder at all."
    )
    buffer = io.BytesIO()
    prs.save(buffer)

    doc = ingest_uploaded_bytes(
        kb_id="kb_v4", source=user_source("untitled.pptx", len(buffer.getvalue())),
        upload_dir=tmp_path, file_name="untitled.pptx", data=buffer.getvalue(),
    )
    text = Path(doc.file_path).read_text(encoding="utf-8")
    chunks = get_chunker("section-aware").chunk(
        document=doc, text=text, source=user_source("untitled.pptx"), target_size=400, overlap=40
    )
    assert chunks
    assert chunks[0].slide == 1
    assert chunks[0].slide_title is None
    assert "# Slide 1" not in text


def test_legacy_ppt_magic_bytes_are_accepted_then_rejected_by_parser():
    ole2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64
    assert validate_upload("old.ppt", ole2).ok


def test_ooxml_part_check_rejects_plain_zip(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("hello.txt", "not an office document")
    result = validate_upload("notes.docx", buffer.getvalue())
    assert not result.ok
    assert "word/document.xml" in result.error


# ---------------------------------------------------------------------------
# Phase 3: invalid documents never silently succeed
# ---------------------------------------------------------------------------

def test_corrupt_pptx_bytes_raise_ingestion_error(tmp_path):
    source = user_source("broken.pptx", 100)
    bad = b"PK\x03\x04" + b"garbage" * 40
    with pytest.raises(IngestionError):
        ingest_uploaded_bytes(
            kb_id="kb_v4", source=source, upload_dir=tmp_path, file_name="broken.pptx", data=bad
        )


def test_corrupt_docx_bytes_raise_ingestion_error(tmp_path):
    source = Source(
        id=new_id("src"), kb_id="kb_v4", url="upload://b.docx", title="b.docx",
        source_type=SourceType.DOCUMENT, user_provided=True,
    )
    with pytest.raises(IngestionError):
        ingest_uploaded_bytes(
            kb_id="kb_v4", source=source, upload_dir=tmp_path,
            file_name="b.docx", data=b"PK\x03\x04not really a docx",
        )


def test_image_only_pdf_is_reported_as_unextractable(tmp_path):
    """No OCR is implemented — say so rather than returning an empty document."""
    empty_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 2 >>\nstream\n\n\nendstream\nendobj\n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\n%%EOF\n"
    )
    source = Source(
        id=new_id("src"), kb_id="kb_v4", url="upload://scan.pdf", title="scan.pdf",
        source_type=SourceType.PDF, user_provided=True,
    )
    with pytest.raises(IngestionError):
        ingest_uploaded_bytes(
            kb_id="kb_v4", source=source, upload_dir=tmp_path,
            file_name="scan.pdf", data=empty_pdf,
        )


# ---------------------------------------------------------------------------
# Phase 4: user-provided integrity model
# ---------------------------------------------------------------------------

def test_user_scorer_never_penalises_missing_authority_signals():
    assessment = UserProvidedIntegrityScorer().assess(
        file_valid=True, parse_ok=True, extracted_chars=5000, unit_count=20
    )
    assert assessment.decision.value == "ACCEPT"
    assert assessment.score > 0.85
    # No authority/recency/accessibility/evidence weight is used at all.
    assert set(assessment.weights) == {
        "file_validity", "content_extraction", "user_relevance", "duplication", "structure",
    }
    assert assessment.signals.authority == 0.5, "authority is not applicable, not zero"
    assert assessment.signals.recency == 0.5
    assert any("asserted by the uploader" in r for r in assessment.reasons)


def test_user_scorer_rejects_only_hard_technical_failures():
    corrupt = assess_user_upload(file_valid=False, parse_ok=False, extracted_chars=0)
    assert corrupt.decision.value == "REJECT"
    assert any("failed validation" in r for r in corrupt.reasons)

    no_text = assess_user_upload(file_valid=True, parse_ok=True, extracted_chars=0)
    assert no_text.decision.value == "REJECT"
    assert any("OCR" in w for w in no_text.warnings)

    # A short class note is legitimate: it is accepted with a warning, never
    # downgraded for being thin.
    thin = assess_user_upload(file_valid=True, parse_ok=True, extracted_chars=50)
    assert thin.decision.value == "ACCEPT"
    assert thin.signals.content_extraction < 0.5
    assert any("Very little text" in w for w in thin.warnings)

    # A file that validated but whose parser could not read it needs review.
    unreadable = assess_user_upload(
        file_valid=True, parse_ok=False, extracted_chars=0, parse_error="unsupported structure"
    )
    assert unreadable.decision.value == "REVIEW"
    assert any("unsupported structure" in w for w in unreadable.warnings)


def test_user_scorer_reports_duplicates_without_rejecting():
    dup = assess_user_upload(
        file_valid=True, parse_ok=True, extracted_chars=9000, duplicate_of="doc_existing"
    )
    assert dup.decision.value == "ACCEPT"
    assert dup.signals.duplication == 0.0
    assert any("already exists" in r for r in dup.reasons)
    assert any("nothing was replaced" in w for w in dup.warnings)


def test_user_scorer_weights_sum_to_one():
    from app.services.source_quality.user_scorer import USER_WEIGHTS

    assert abs(sum(USER_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Provenance completeness on every supported upload format
# ---------------------------------------------------------------------------

def test_every_chunk_preserves_the_full_provenance_set(tmp_path):
    gm = "The metacentric height GM equals KB plus BM minus KG, measured vertically."
    cases = [
        ("deck.pptx", make_pptx([("Stability", [gm]), ("Trim", ["Trim balances the longitudinal moments of the ship."])])),
        ("notes.docx", make_docx([("heading", "Stability"), ("body", gm), ("heading", "Trim"), ("body", "Trim balances the longitudinal moments of the ship.")])),
        ("paper.pdf", make_pdf_like_bytes()),
        ("readme.md", ("# Stability\n\n" + gm + "\n").encode()),
        ("plain.txt", ("GM is the metacentric height of a floating body in still water.\n").encode()),
    ]
    for name, data in cases:
        source = user_source(name, len(data))
        doc = ingest_uploaded_bytes(
            kb_id="kb_v4", source=source, upload_dir=tmp_path, file_name=name, data=data
        )
        text = Path(doc.file_path).read_text(encoding="utf-8")
        chunks = get_chunker("section-aware").chunk(
            document=doc, text=text, source=source, target_size=400, overlap=50,
            domain="Naval Architecture", chunking_config={"target_size": 400, "overlap": 50},
        )
        assert chunks, f"{name} produced no chunks"
        for chunk in chunks:
            assert chunk.document_id == doc.id
            assert chunk.source_id == source.id
            assert chunk.content_hash
            assert chunk.id.startswith("chk_")
            assert chunk.chunk_index >= 0
            assert chunk.chunking_strategy == "section-aware"
            assert chunk.chunking_config["target_size"] == 400
            assert chunk.document_version == 1
            assert chunk.user_provided is True
        if name.endswith(".pptx"):
            assert all(c.slide is not None for c in chunks)
            assert all(c.page is None for c in chunks)
        if name.endswith(".pdf"):
            assert all(c.page is not None for c in chunks)
            assert all(c.slide is None for c in chunks)


def test_text_parser_class_is_still_exported_for_backwards_compatibility():
    assert TextParser.extensions == (".txt",)
    assert MarkdownParser.extensions == (".md", ".markdown", ".mdown")
    assert PdfParser.extensions == (".pdf",)
    assert set(PARSER_REGISTRY.names()) >= {"pdf", "pptx", "docx", "html", "txt", "markdown"}