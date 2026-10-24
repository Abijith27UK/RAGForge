import sys
sys.path.insert(0, '.')
from app.repo.sqlite_repo import Repository
repo = Repository()
kbs = repo.list_kbs()
for kb in kbs:
    print(f"KB: {kb.id} - name: {kb.name} - domain: {kb.domain} - purpose: {kb.purpose}")
    print(f"  depth: {kb.depth} - status: {kb.status} - chunks: {repo.count_chunks(kb.id)} - docs: {repo.count_documents(kb.id)}")
    print(f"  vector_backend: {kb.vector_backend}")
    print(f"  embedding_identity: {kb.embedding_identity}")
    print()