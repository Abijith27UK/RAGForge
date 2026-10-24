
from pathlib import Path

from app.repositories.sqlite_repo import Repository
from app.services.answer_eval.retrieval_diagnostic_runner import (
    RetrievalDiagnosticRunner,
)
from app.config import get_settings


def main():
    settings = get_settings()

    # Find the database path configured by your application.
    db_path = next(
        (
            getattr(settings, name)
            for name in (
                "db_path",
                "sqlite_db_path",
                "database_path",
                "sqlite_path",
                "sqlite_database_path",
            )
            if getattr(settings, name, None)
        ),
        None,
    )

    if not db_path:
        raise RuntimeError(
            "Database path not found in settings. "
            "Check app/config.py and sqlite_repo.py."
        )

    # Resolve relative database paths from the backend directory.
    db_path = Path(db_path)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parent / db_path

    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")

    repo = Repository(db_path=str(db_path))
    runner = RetrievalDiagnosticRunner(repo, settings)

    report = runner.run(
        kb_id="kb_f278c283c748",
        strategies=["dense", "bm25", "hybrid"],
        top_k=10,
        persist=True,
    )

    print("\n========== RETRIEVAL DIAGNOSTIC REPORT ==========")
    print(report)
    print("=================================================")


if __name__ == "__main__":
    main()
