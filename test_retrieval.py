import sys
sys.path.insert(0, 'backend')

from app.services.retrieval.service import RetrievalService
from app.repositories.sqlite_repo import Repository
from app.config import get_settings

settings = get_settings()
repo = Repository('backend/data/ragforge.db')
service = RetrievalService(repo, settings)

# Test a simple query about LNG
response = service.run('kb_b554bfde9d7f', 'LNG temperature', strategy='dense', overrides={'top_k': 3})
print('Results for "LNG temperature":')
for i, r in enumerate(response.results):
    prov = r.provenance if isinstance(r.provenance, dict) else r.provenance.model_dump()
    print(f'  {i+1}. chunk_id={r.chunk_id}, doc_id={r.document_id}, score={r.score:.4f}')
    print(f'     section={prov.get("section", "N/A")}, page={prov.get("page", "N/A")}')
    print(f'     source_title={prov.get("source_title", "N/A")}')
    print(f'     method={r.retrieval_method}')
    print()

# Test the specific query from the task
print('=\"\"\"=\"\"\"=\"\"\"=\"\"\"=\"\"\"=\"\"\"=\"\"\"')
print('Query: "What are the differences between LNG, CNG, PNG and LPG?"')
print('=\"\"\"=\"\"\"=\"\"\"=\"\"\"=\"\"\"=\"\"\"=\"\"\"')
response2 = service.run('kb_b554bfde9d7f', 'What are the differences between LNG, CNG, PNG and LPG?', strategy='dense', overrides={'top_k': 5})
print('Results:')
for i, r in enumerate(response2.results):
    prov = r.provenance if isinstance(r.provenance, dict) else r.provenance.model_dump()
    print(f'  {i+1}. chunk_id={r.chunk_id}, doc_id={r.document_id}, score={r.score:.4f}')
    print(f'     section={prov.get("section", "N/A")}, page={prov.get("page", "N/A")}')
    print(f'     source_title={prov.get("source_title", "N/A")}')
    print(f'     method={r.retrieval_method}')
    print(f'     text_excerpt={r.text[:120] if r.text else "N/A"}...')
    print()