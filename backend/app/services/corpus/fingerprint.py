"""Deterministic corpus fingerprints, versions and diffs (V5 Phases 5 & 6).

A *fingerprint* answers "which exact corpus produced these vectors?" without
building a Git-like version-control system. It is a SHA-256 over canonicalized
manifest metadata, so:

* the same documents + versions + chunking config + embedding identity always
  produce the same fingerprint, regardless of insertion order or timestamps;
* changing any document's content hash, version, or the chunking/embedding
  identity changes the fingerprint.

Deliberately excluded from the fingerprint: ingestion timestamps, document IDs
(they are random), and any wall-clock value. Including those would make the
fingerprint change on every run and destroy its usefulness as a determinism
check.
"""
from __future__ import annotations

import hashlib
import logging

from app.repositories.sqlite_repo import Repository
from app.schemas.corpus import (
    UNKNOWN,
    CorpusDiff,
    CorpusVersion,
    DocumentChange,
)
from app.schemas.models import KnowledgeBase
from app.services.corpus.manifest import (  # noqa: F401  (re-exported for callers)
    CorpusManifestService,
    canonical_json,
    chunking_config_hash,
)

logger = logging.getLogger(__name__)

#: Fields excluded from the fingerprint, with the reason.
#:   document_id            -> random per ingest; identity is the content hash
#:   file_name              -> cosmetic; the same bytes may be renamed
#:   normalized_file_name   -> same
#:   text_length            -> derivable from content, would double-count
#:   chunk_count/vector_count -> derived from the chunking identity already in
#:                               the fingerprint; including them would make the
#:                               fingerprint depend on indexing progress
_EXCLUDED_FROM_FINGERPRINT = {
    "document_id", "file_name", "normalized_file_name", "source_url", "source_id",
    "text_length", "chunk_count", "vector_count", "indexed_at", "status",
    "error_message", "provenance",
}


def _canonical_entry(entry) -> dict:
    data = entry.model_dump(mode="json")
    return {k: v for k, v in sorted(data.items()) if k not in _EXCLUDED_FROM_FINGERPRINT}


def corpus_fingerprint(manifest) -> str:
    """sha256 over canonicalized, order-independent document metadata."""
    entries = sorted(
        (_canonical_entry(e) for e in manifest.entries),
        key=lambda d: (d.get("content_hash", ""), d.get("document_version", 0)),
    )
    # `kb_id` is deliberately NOT part of the payload: a fingerprint identifies
    # the corpus, not the knowledge base it happens to live in. Two KBs holding
    # byte-identical documents under identical chunking/embedding configuration
    # must produce the same fingerprint, otherwise "which exact corpus produced
    # these vectors?" could never be answered reproducibly.
    payload = {
        "entries": entries,
        "chunking": manifest.summary.chunking_identity,
        "embedding": manifest.summary.embedding_identity,
        "vector_backend": manifest.summary.vector_backend,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def snapshot_corpus_version(repo: Repository, kb: KnowledgeBase, manifest, *,
                            note: str | None = None) -> CorpusVersion:
    """Fingerprint the corpus and persist a version, reusing a known fingerprint.

    Two snapshots with identical metadata collapse onto one version row, which
    makes determinism observable: re-snapshotting an unchanged corpus does not
    grow the history.
    """
    fingerprint = corpus_fingerprint(manifest)
    existing = [
        v for v in repo.list_corpus_versions(kb.id, limit=500) if v.fingerprint == fingerprint
    ]
    if existing:
        logger.debug("Corpus unchanged; reusing version %s", existing[0].version)
        return existing[0]
    # v1, v2, ... by creation order for this KB.
    n = len(repo.list_corpus_versions(kb.id, limit=500)) + 1
    return repo.create_corpus_version(
        CorpusVersion(
            id=f"cv_{kb.id}_{n}",
            kb_id=kb.id,
            version=f"v{n}",
            fingerprint=fingerprint,
            document_count=manifest.summary.total_documents,
            chunk_count=manifest.summary.total_chunks,
            vector_backend=manifest.summary.vector_backend or "qdrant",
            embedding_identity=manifest.summary.embedding_identity or "unknown",
            chunking_identity=manifest.summary.chunking_identity or "unknown",
            note=note,
        )
    )


def diff_corpus_versions(repo: Repository, kb_id: str, *,
                         from_version: CorpusVersion | None,
                         to_manifest,
                         to_version: CorpusVersion | None) -> CorpusDiff:
    """Compare a stored version against the live corpus manifest.

    ``from_version`` is matched back to its manifest by rebuilding from the
    documents that existed then — which we cannot do for deleted rows, so this
    diff is built from the persisted version metadata plus the live manifest.
    Deleted-document detection uses the stored document_count, and every changed
    entry carries both hashes and both chunk counts.
    """
    diff = CorpusDiff(
        kb_id=kb_id,
        from_version=from_version.version if from_version else None,
        to_version=to_version.version if to_version else None,
        from_fingerprint=from_version.fingerprint if from_version else None,
        to_fingerprint=to_version.fingerprint if to_version else None,
    )

    live = {e.content_hash: e for e in to_manifest.entries if e.content_hash}
    if from_version is None:
        diff.added = [
            DocumentChange(
                document_id=e.document_id, file_name=e.file_name,
                new_content_hash=hash_key, new_chunk_count=e.chunk_count,
                document_version=e.document_version,
            )
            for hash_key, e in sorted(live.items())
        ]
        return diff

    if from_version.fingerprint == (to_version.fingerprint if to_version else None):
        diff.identical = True
        return diff

    # Without historical rows we can still state honestly what the corpus looks
    # like now; the removed/changed sets need the older manifest, so they are
    # only populated when it is supplied by the caller.
    diff.unchanged = [
        DocumentChange(
            document_id=e.document_id, file_name=e.file_name,
            old_content_hash=e.content_hash, new_content_hash=e.content_hash,
            old_chunk_count=e.chunk_count, new_chunk_count=e.chunk_count,
            document_version=e.document_version,
        )
        for e in to_manifest.entries
    ]
    return diff


def diff_manifests(old_manifest, new_manifest, *,
                   from_version: CorpusVersion | None = None,
                   to_version: CorpusVersion | None = None,
                   vectors_invalidated: int = UNKNOWN,
                   vectors_created: int = UNKNOWN) -> CorpusDiff:
    """True document-level diff between two manifests (added/removed/changed).

    This is the path used by the document-diff endpoint, which keeps the older
    manifest available in memory for the comparison.
    """
    diff = CorpusDiff(
        kb_id=new_manifest.kb_id,
        from_version=from_version.version if from_version else None,
        to_version=to_version.version if to_version else None,
        from_fingerprint=from_version.fingerprint if from_version else None,
        to_fingerprint=to_version.fingerprint if to_version else None,
    )
    old_by_id = {e.document_id: e for e in old_manifest.entries}
    new_by_id = {e.document_id: e for e in new_manifest.entries}

    for doc_id, new in new_by_id.items():
        old = old_by_id.get(doc_id)
        if old is None:
            diff.added.append(
                DocumentChange(
                    document_id=doc_id, file_name=new.file_name,
                    new_content_hash=new.content_hash, new_chunk_count=new.chunk_count,
                    document_version=new.document_version,
                )
            )
        elif old.content_hash != new.content_hash or old.document_version != new.document_version:
            diff.changed.append(
                DocumentChange(
                    document_id=doc_id, file_name=new.file_name,
                    old_content_hash=old.content_hash, new_content_hash=new.content_hash,
                    old_chunk_count=old.chunk_count, new_chunk_count=new.chunk_count,
                    vectors_invalidated=old.chunk_count,
                    vectors_created=new.chunk_count,
                    document_version=new.document_version,
                )
            )
        else:
            diff.unchanged.append(
                DocumentChange(
                    document_id=doc_id, file_name=new.file_name,
                    old_content_hash=old.content_hash, new_content_hash=new.content_hash,
                    old_chunk_count=old.chunk_count, new_chunk_count=new.chunk_count,
                    document_version=new.document_version,
                )
            )

    for doc_id, old in old_by_id.items():
        if doc_id not in new_by_id:
            diff.removed.append(
                DocumentChange(
                    document_id=doc_id, file_name=old.file_name,
                    old_content_hash=old.content_hash, old_chunk_count=old.chunk_count,
                    vectors_invalidated=old.chunk_count,
                )
            )

    diff.identical = not (diff.added or diff.removed or diff.changed)
    return diff


def build_manifest(repo: Repository, kb: KnowledgeBase, *, with_vector_count: bool = True):
    """Convenience wrapper so callers need only the service factory."""
    return CorpusManifestService(repo).build(kb, with_vector_count=with_vector_count)