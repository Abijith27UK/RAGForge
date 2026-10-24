import sys
import os
sys.path.insert(0, '.')
from app.repo.sqlite_repo import Repository
from app.config import get_settings
settings = get_settings()
db_path = settings.db_path
print(f"DB path: {db_path}")
repo = Repository(str)
kbs = repo.list_kbs()
for kb in kbs:
    print(f'KB: {kb.id} - {kb.name} - domain: {kb.domain} - status: {kb.status} - chunks: {repo.count_chunks(kb.id)} - docs: {repo.count_documents(kb.id)}')