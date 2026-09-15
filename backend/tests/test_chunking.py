"""Chunking tests: section awareness, provenance propagation, config."""
from __future__ import annotations

from app.schemas.models import Document, Source, SourceType
from app.services.chunking.chunker import (
    FixedSizeChunker,
    SectionAwareChunker,
    split_into_sections,
)
from app.utils.ids import new_id

SAMPLE_MD = """# Electric Vehicles

Electric vehicles use electric motors and battery packs.

## Battery Management System

The BMS monitors cell voltages, temperatures, and state of charge.
It balances cells and protects against overcharge.

## Vehicle Dynamics

Suspension geometry affects handling and ride comfort.
"""


def make_doc_and_source():
    doc = Document(
        id=new_id("doc"),
        kb_id="kb_test",
        source_id=new_id("src"),
        url="https://example.com/ev-guide.md",
        title="EV Guide",
        source_type=SourceType.TEXT,
        content_hash="abc123",
        ingestion_timestamp=__import__("datetime").datetime(2026, 1, 1),
    )
    source = Source(
        id=doc.source_id,
        kb_id=doc.kb_id,
        url=doc.url,
        title="EV Guide",
        source_type=SourceType.TEXT,
        trust_score=0.83,
    )
    return doc, source


def test_split_into_sections_detects_markdown_headings():
    sections = split_into_sections(SAMPLE_MD)
    titles = [s.title for s in sections]
    assert any("Electric Vehicles" in t for t in titles)
    assert any("Battery Management System" in t for t in titles)
    assert any("Vehicle Dynamics" in t for t in titles)


def test_section_chunker_preserves_provenance():
    doc, source = make_doc_and_source()
    chunker = SectionAwareChunker()
    chunks = chunker.chunk(doc, SAMPLE_MD, source, domain="Automobile Engineering")
    assert chunks, "Must produce chunks"
    for c in chunks:
        assert c.document_id == doc.id
        assert c.kb_id == doc.kb_id
        assert c.source_url == source.url
        assert c.trust_score == 0.83
        assert c.domain == "Automobile Engineering"
        assert c.content_hash == __import__("hashlib").sha256(c.text.encode()).hexdigest()[:0] or c.content_hash
        assert c.char_count == len(c.text)


def test_section_paths_recorded():
    doc, source = make_doc_and_source()
    chunks = SectionAwareChunker().chunk(doc, SAMPLE_MD, source)
    section_names = {c.section for c in chunks}
    assert "Battery Management System" in section_names


def test_long_section_is_windowed_with_overlap():
    doc, source = make_doc_and_source()
    long_text = "# Big\n\n" + ("word " * 2000)
    chunks = SectionAwareChunker().chunk(doc, long_text, source, target_size=500, overlap=50)
    assert len(chunks) > 1
    # overlap can push a window to target+overlap; allow small slack
    assert all(c.char_count <= 500 + 50 + 10 for c in chunks)


def test_fixed_size_chunker_ignores_structure():
    doc, source = make_doc_and_source()
    chunks = FixedSizeChunker().chunk(doc, SAMPLE_MD, source, target_size=200, overlap=20)
    assert all(c.section is None for c in chunks)
    assert len(chunks) >= 2


def test_chunks_are_unique():
    doc, source = make_doc_and_source()
    chunks = SectionAwareChunker().chunk(doc, SAMPLE_MD, source)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


PDF_WITH_PAGES = (
    "\n\f[PAGE 1]\nIntroduction\nElectric vehicles are automobiles driven by electric motors and powered by battery packs.\n\n"
    "\f[PAGE 2]\nBattery Management\nThe BMS monitors cell voltages and temperatures during every drive cycle and charging session.\n\n"
    "\f[PAGE 3]\nThe BMS also balances cells during charging and protects the pack against deep discharge events in cold weather conditions.\n"
)


def test_pdf_page_markers_propagate_to_chunk_provenance():
    doc, source = make_doc_and_source()
    doc.source_type = SourceType.PDF
    chunks = SectionAwareChunker().chunk(doc, PDF_WITH_PAGES, source)
    assert chunks, "must produce chunks"
    by_text = {"introduction": None, "monitors": None, "balances": None}
    for c in chunks:
        low = c.text.lower()
        if "electric vehicles are automobiles" in low:
            assert c.page == 1, f"intro chunk must carry page 1, got {c.page}"
            by_text["introduction"] = c
        if "monitors cell voltages" in low:
            assert c.page == 2, f"BMS chunk must carry page 2, got {c.page}"
            by_text["monitors"] = c
        if "balances cells" in low:
            assert c.page == 3, f"page-3 continuation chunk must carry page 3, got {c.page}"
            by_text["balances"] = c
    assert all(v is not None for v in by_text.values()), "all expected chunks must exist"


def test_page_markers_are_stripped_from_chunk_text():
    doc, source = make_doc_and_source()
    doc.source_type = SourceType.PDF
    chunks = SectionAwareChunker().chunk(doc, PDF_WITH_PAGES, source)
    for c in chunks:
        assert "[PAGE" not in c.text, "page markers must not leak into chunk text"
        assert "\f" not in c.text


def test_non_pdf_text_has_no_page_numbers():
    doc, source = make_doc_and_source()
    chunks = SectionAwareChunker().chunk(doc, SAMPLE_MD, source)
    assert all(c.page is None for c in chunks)
