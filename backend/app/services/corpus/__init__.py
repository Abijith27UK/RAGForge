"""Corpus-engineering services: bulk ingestion, manifest, integrity, repair.

The product direction is a corpus-engineering platform, not a chatbot. These
services make the corpus itself observable and repairable.
"""
from app.services.corpus.batches import IngestionBatchService, item_key_for
from app.services.corpus.fingerprint import (
    build_manifest,
    corpus_fingerprint,
    diff_corpus_versions,
    diff_manifests,
    snapshot_corpus_version,
)
from app.services.corpus.integrity import CorpusIntegrityService
from app.services.corpus.manifest import CorpusManifestService, chunking_config_hash
from app.services.corpus.repair import CorpusRepairService

__all__ = [
    "CorpusIntegrityService",
    "CorpusManifestService",
    "CorpusRepairService",
    "IngestionBatchService",
    "build_manifest",
    "chunking_config_hash",
    "corpus_fingerprint",
    "diff_corpus_versions",
    "diff_manifests",
    "item_key_for",
    "snapshot_corpus_version",
]