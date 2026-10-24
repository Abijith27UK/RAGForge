#!/usr/bin/env python3
"""Setup script for OE6008 test knowledge base.

This script creates an OE6008 knowledge base, uploads the course documents,
and runs the chunking/embedding/indexing pipeline.

Usage:
    python setup_oe6008_kb.py

Requirements:
    - Qdrant running at http://127.0.0.1:6333
    - Backend API running at http://localhost:8000
    - OE6008 PDF files in the working directory
"""

import json
import os
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.config import get_settings, Settings
from app.repositories.sqlite_repo import Repository
from app.services.ingestion.ingestion import ingest_uploaded_bytes
from app.services.ingestion.upload import sanitize_file_name, validate_upload
from app.services.ingestion.parsers import supported_upload_extensions
from app.services.source_quality.user_scorer import assess_user_upload
from app.utils.ids import new_id
from app.utils.text import sha256_bytes
from app.services.embeddings.provider import create_embedding_provider
from app.services.vector_store.factory import create_vector_store
from app.services.indexing.document_indexer import index_documents
from app.schemas.models import DocumentStatus, KBStatus, Source, SourceType
from datetime import datetime, timezone

# OE6008 documents
OE6008_DIR = Path(__file__).parent / "OE6008"
DATA_DIR = Path(__file__).parent / "backend" / "data"


def create_test_kb():
    """Create an empty OE6008 knowledge base."""
    settings = get_settings()
    repo = Repository(str(settings.db_path))
    
    # Check if OE6008 KB already exists
    kb = repo.get_kb("OE6008")
    if kb:
        print("OE6008 KB already exists, using existing one")
        return kb
    
    # Create new KB
    kb_id = new_id("kb")
    from app.schemas.models import KnowledgeBase
    kb = KnowledgeBase(
        id=kb_id,
        name="OE6008",
        domain="Education",
        purpose="OE6008 Course Knowledge Base",
        target_audience="Students",
        depth="intermediate",
        status=KBStatus.CREATED,
        version=1,
    )
    repo.create_kb(kb)
    print(f"Created OE6008 KB with id={kb_id}")
    return kb


def upload_documents(kb_id):
    """Upload all 6 OE6008 PDF documents."""
    settings = get_settings()
    repo = Repository(str(settings.db_path))
    
    documents_dir = OE6008_DIR
    pdf_files = sorted(documents_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to upload")
    
    uploaded_docs = []
    
    for pdf_path in pdf_files:
        file_name = pdf_path.name
        print(f"\nUploading: {file_name}")
        
        # Read file data
        data = pdf_path.read_bytes()
        size = len(data)
        extension = pdf_path.suffix.lower()
        
        # Validate upload
        validation = validate_upload(
            file_name, data, 
            max_bytes=settings.max_upload_bytes,
            allowed_extensions=supported_upload_extensions(),
        )
        
        if not validation.ok:
            print(f"  REJECTED: {validation.error}")
            continue
        
        # Check for duplicate content
        content_hash = sha256_bytes(data)
        existing = repo.find_document_by_hash(kb_id, content_hash)
        if existing is not None:
            print(f"  DUPLICATE: content already exists (doc id={existing.id})")
            # Use existing document
            doc = existing
            doc.status = DocumentStatus.READY
            repo.update_document(doc)
            uploaded_docs.append(doc)
            continue
        
        # Create source
        source = Source(
            id=new_id("src"),
            kb_id=kb_id,
            url=f"upload://{file_name}",
            title=file_name,
            source_type=SourceType.PDF,
            publisher=None,
            discovered_via="user-upload",
            user_provided=True,
            provenance="user_upload",
            file_name=file_name,
            file_size=size,
        )
        repo.create_source(source)
        
        # Ingest bytes
        try:
            doc = ingest_uploaded_bytes(
                kb_id=kb_id,
                source=source,
                upload_dir=settings.kb_upload_dir(kb_id),
                file_name=file_name,
                data=data,
                mime_type=None,  # Will be detected from extension
            )
            
            # Assess the upload
            assessment = assess_user_upload(
                file_valid=True,
                parse_ok=True,
                extracted_chars=doc.text_length,
                unit_count=doc.page_count or doc.slide_count or doc.section_count or 0,
                warnings=validation.warnings,
            )
            source.trust_score = assessment.score
            source.decision = assessment.decision
            source.integrity = assessment.model_dump(mode="json")
            source.quality = assessment.model_dump(mode="json")
            repo.update_source(source)
            
            # Update document status
            doc.status = DocumentStatus.PARSED if assessment.decision.value != "REJECT" else DocumentStatus.FAILED
            repo.create_document(doc) if not repo.document_exists(kb_id, doc.id) else repo.update_document(doc)
            
            print(f"  OK: parsed {doc.text_length} chars, {assessment.decision.value} decision")
            uploaded_docs.append(doc)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            # Create a failed document record
            doc = type('Doc', (), {})()
            doc.id = new_id("doc")
            doc.text_length = 0
            doc.status = DocumentStatus.FAILED
            doc.parse_error = str(e)
            doc.file_name = file_name
            doc.user_provided = True
            repo.create_document(doc)
            uploaded_docs.append(doc)
    
    # Update KB status
    kb = repo.get_kb(kb_id)
    if kb and uploaded_docs:
        kb.status = KBStatus.BUILDING
        kb.updated_at = datetime.now(timezone.utc)
        repo.update_kb(kb)
    
    return uploaded_docs


def index_documents(kb_id, docs):
    """Chunk, embed and index the uploaded documents."""
    settings = get_settings()
    repo = Repository(str(settings.db_path))
    
    # Filter to only successfully parsed documents
    valid_docs = [d for d in docs if d.status == DocumentStatus.PARSED]
    
    if not valid_docs:
        print("No valid documents to index (all failed parsing)")
        return False
    
    print(f"\nIndexing {len(valid_docs)} documents...")
    
    try:
        # Create embedding provider
        embedder = create_embedding_provider(settings)
        
        # Create vector store
        store = create_vector_store(settings, backend=(repo.get_kb(kb_id).vector_backend or "qdrant"))
        
        # Index documents
        outcome = index_documents(
            repo=repo,
            kb=repo.get_kb(kb_id),
            documents=valid_docs,
            store=store,
            embedder=embedder,
            chunker_name="section-aware",
            target_size=1200,
            overlap=150,
            full_build=False,
            kb_version=repo.get_kb(kb_id).version + 1,
        )
        
        # Update KB status to READY
        kb = repo.get_kb(kb_id)
        kb.version += 1
        kb.last_build_at = datetime.now(timezone.utc)
        kb.updated_at = kb.last_build_at
        kb.status = KBStatus.READY
        repo.update_kb(kb)
        
        print(f"Indexing complete: {outcome['chunk_count']} chunks indexed")
        print(f"KB version: {kb.version}, status: {kb.status}")
        return True
        
    except Exception as e:
        print(f"Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        # Reset KB status
        kb = repo.get_kb(kb_id)
        if kb:
            kb.status = KBStatus.BUILDING
            repo.update_kb(kb)
        return False


def verify_kb(kb_id):
    """Verify the knowledge base is ready and functional."""
    from app.config import get_settings
    from app.repositories.sqlite_repo import Repository
    
    settings = get_settings()
    repo = Repository(str(settings.db_path))
    kb = repo.get_kb(kb_id)
    
    if not kb:
        print(f"KB {kb_id} not found")
        return False
    
    print(f"\n=== KB Verification for {kb_id} ===")
    print(f"Name: {kb.name}")
    print(f"Domain: {kb.domain}")
    print(f"Purpose: {kb.purpose}")
    print(f"Target audience: {kb.target_audience}")
    print(f"Depth: {kb.depth}")
    print(f"Status: {kb.status}")
    print(f"Version: {kb.version}")
    
    chunk_count = repo.count_chunks(kb_id)
    document_count = repo.count_documents(kb_id)
    print(f"Chunks: {chunk_count}")
    print(f"Documents: {document_count}")
    
    # Check vector store
    try:
        store = create_vector_store(settings, backend=(kb.vector_backend or "qdrant"))
        info = store.collection_info(kb_id)
        points = info.get("points_count", 0)
        collection = info.get("name", "unknown")
        print(f"Vector store points: {points} (collection: {collection})")
        print(f"Points match chunks: {points == chunk_count}")
    except Exception as e:
        print(f"Vector store not reachable: {e}")
    
    # Check embedding identity
    identity = kb.embedding_identity
    if identity:
        print(f"Embedding identity: {identity}")
    else:
        print("No embedding identity recorded")
    
    # Validate knowledge base
    try:
        from app.services.evaluation.quality_audit import QualityAuditService
        audit = QualityAuditService(repo, settings)
        report = audit.run_audit(kb_id)
        print(f"\nQuality audit: {report.get('status', 'unknown')}")
        print(f"  Issues: {len(report.get('issues', []))}")
        for issue in report.get('issues', [])[:5]:
            print(f"    - {issue}")
    except Exception as e:
        print(f"Quality audit error: {e}")
    
    # Check retrieval
    try:
        from app.services.retrieval.service import RetrievalService
        retrieval = RetrievalService(repo, settings)
        
        # Try a simple search
        response = retrieval.run(kb_id, "test query", strategy="dense", overrides={"top_k": 3})
        print(f"\nRetrieval test: {len(response.results)} results")
        if response.results:
            r = response.results[0]
            prov = r.provenance or {}
            print(f"  Top result: chunk={r.chunk_id}, doc={r.document_id}")
            print(f"  Score: {r.score:.4f}")
            print(f"  Text (first 80 chars): {r.text[:80] if r.text else 'empty'}")
            print(f"  Provenance: {json.dumps(prov, indent=2)[:200]}")
    except Exception as e:
        print(f"Retrieval test error: {e}")
    
    return kb.status == KBStatus.READY and chunk_count > 0


def main():
    """Main entry point."""
    print("=" * 60)
    print("RAGForge OE6008 Test Knowledge Base Setup")
    print("=" * 60)
    
    # Step 1: Create KB
    print("\n[1/5] Creating OE6008 knowledge base...")
    kb = create_test_kb()
    kb_id = kb.id
    
    # Step 2: Upload documents
    print("\n[2/5] Uploading OE6008 documents...")
    uploaded_docs = upload_documents(kb_id)
    
    # Step 3: Index documents
    print("\n[3/5] Indexing documents (chunking + embedding + vector store)...")
    success = index_documents(kb_id, uploaded_docs)
    
    if not success:
        print("WARNING: Indexing may have had issues, but continuing verification...")
    
    # Step 4: Verify KB
    print("\n[4/5] Verifying knowledge base...")
    ready = verify_kb(kb_id)
    
    # Step 5: Summary
    print("\n[5/5] Summary")
    print("=" * 60)
    if ready:
        print("OE6008 KB is ready for retrieval and questioning!")
    else:
        print("OE6008 KB created but may have issues - check output above")
    
    print("=" * 60)


if __name__ == "__main__":
    main()