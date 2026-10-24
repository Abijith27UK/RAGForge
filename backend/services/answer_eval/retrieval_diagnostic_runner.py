
from app.repositories.sqlite_repo import Repository
from app.services.answer_eval.retrieval_diagnostic_runner import (
    RetrievalDiagnosticRunner,
)
from app.config import get_settings


def main():
    repo = Repository()
    runner = RetrievalDiagnosticRunner(repo, get_settings())

    report = runner.run(
        kb_id="kb_f278c283c748",
        strategies=["dense", "bm25", "hybrid"],
        top_k=10,
        persist=True,
    )

    print(report)


if __name__ == "__main__":
    main()
