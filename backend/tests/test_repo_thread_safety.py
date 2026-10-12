"""Repository thread-safety: ONE SQLite connection used by many request threads.

Why this test exists, in the project's own terms:

* The API wires a single ``Repository`` (``app.api.deps.get_repo`` is cached), so
  every request in the process shares one ``sqlite3.Connection``.
* FastAPI runs synchronous endpoints in a thread pool, so several requests can
  reach that connection at the same time. ``check_same_thread=False`` makes that
  legal for sqlite3 only if the repository serialises the calls itself.
* It did not, and the failure was observed on the held-out review page, which
  fires several requests at once:

      sqlite3.InterfaceError: bad parameter or other API misuse
      -> 500 Internal Server Error -> the page rendered "Knowledge base not found"

These tests drive that pattern directly — many threads released together by a
barrier, mixing reads and writes against one shared repository — so the guarantee
is pinned by measurement rather than by inspection. They assert on real state
afterwards as well, so a "fix" that serialises by dropping writes cannot pass.
"""

from __future__ import annotations

import threading
from typing import Callable

from app.repositories.sqlite_repo import Repository
from app.schemas.models import KnowledgeBase
from app.utils.ids import new_id

THREADS = 16
ROUNDS = 25
JOIN_TIMEOUT = 60.0


def _kb(name: str) -> KnowledgeBase:
    return KnowledgeBase(
        id=new_id("kb"),
        name=name,
        domain="Automobile Engineering",
        purpose="Engineering education",
        target_audience="Engineering students",
    )


def _run_parallel(
    work: Callable[[int], None], threads: int = THREADS
) -> list[BaseException]:
    """Run ``work(i)`` in ``threads`` threads, released at the same instant.

    Every exception is collected instead of being swallowed by the thread, so
    the assertion failure names the sqlite3 error that actually occurred.
    """
    barrier = threading.Barrier(threads)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(index: int) -> None:
        barrier.wait(timeout=JOIN_TIMEOUT)
        try:
            work(index)
        except BaseException as exc:  # noqa: BLE001 - reported on the main thread
            with errors_lock:
                errors.append(exc)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=JOIN_TIMEOUT)
    assert not any(w.is_alive() for w in workers), "a worker thread did not finish"
    return errors


def test_parallel_reads_do_not_break_the_shared_connection(tmp_path):
    """The review page's actual access pattern: concurrent read requests."""
    repo = Repository(str(tmp_path / "test.db"))
    kb = _kb("Concurrent reads")
    repo.create_kb(kb)

    def read(_index: int) -> None:
        for _ in range(ROUNDS):
            assert repo.get_kb(kb.id) is not None
            assert repo.list_kbs()
            assert repo.count_chunks(kb.id) == 0

    errors = _run_parallel(read)
    assert errors == [], f"{len(errors)} thread(s) failed; first: {errors[:1]!r}"


def test_parallel_reads_and_writes_do_not_break_the_shared_connection(tmp_path):
    """Reads and writes at once, which is what a real UI session produces."""
    repo = Repository(str(tmp_path / "test.db"))
    shared = _kb("Concurrent read/write")
    repo.create_kb(shared)

    def mixed(index: int) -> None:
        for round_no in range(ROUNDS):
            repo.create_kb(_kb(f"inserted-{index}-{round_no}"))
            assert repo.get_kb(shared.id) is not None
            repo.update_kb(shared)

    errors = _run_parallel(mixed)
    assert errors == [], f"{len(errors)} thread(s) failed; first: {errors[:1]!r}"

    # The serialisation must not have dropped or corrupted anything.
    stored = repo.list_kbs()
    assert len(stored) == 1 + THREADS * ROUNDS
    assert repo.get_kb(shared.id) is not None


def test_concurrent_review_style_reads_return_the_same_row(tmp_path):
    """Many readers of one row must all see the row, not a torn/failed read."""
    repo = Repository(str(tmp_path / "test.db"))
    kb = _kb("Shared reader")
    repo.create_kb(kb)

    seen: list[str | None] = []
    seen_lock = threading.Lock()

    def read(_index: int) -> None:
        for _ in range(ROUNDS):
            row = repo.get_kb(kb.id)
            with seen_lock:
                seen.append(row.id if row else None)

    errors = _run_parallel(read)
    assert errors == [], f"{len(errors)} thread(s) failed; first: {errors[:1]!r}"
    assert set(seen) == {kb.id}
