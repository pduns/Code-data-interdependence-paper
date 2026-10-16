#!/usr/bin/env python
"""Utilities for parsing folders of PDFs into DOI keyed full-text JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional

import fitz


Parallelism = Literal["none", "thread", "process"]

_DUPLICATE_SUFFIX_RE = re.compile(r" \(\d+\)$")
_DUPLICATE_FILE_RE = re.compile(r" \(\d+\)\.pdf$", flags=re.IGNORECASE)


@dataclass
class PdfParseResult:
    doi: str
    filename: str
    text: Optional[str]
    error: Optional[str]
    page_count: int
    file_size_bytes: int


@dataclass
class ParseRunStats:
    pdf_folder: str
    output_json: str
    failures_json: Optional[str]
    checkpoint_jsonl: str
    manifest_json: Optional[str]
    parallelism: Parallelism
    workers: int
    total_pdf_files: int
    unique_pdf_files: int
    duplicate_pdf_files_removed: int
    queued_pdf_files: int
    skipped_existing: int
    parsed_new: int
    failed_new: int
    checkpoint_write_count: int
    flush_count: int
    flush_elapsed_sec: float
    total_text_chars_new: int
    total_pages_new: int
    elapsed_sec: float


def get_base_name(filename: str) -> str:
    return _DUPLICATE_SUFFIX_RE.sub("", Path(filename).stem)


def filename_to_doi(filename: str) -> str:
    return get_base_name(filename).replace("_", "/")


def has_duplicate_suffix(filename: str) -> bool:
    return bool(_DUPLICATE_FILE_RE.search(filename))


def list_unique_pdf_files(pdf_folder: Path) -> list[Path]:
    all_pdf_files = sorted(
        (path for path in pdf_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda item: item.name.lower(),
    )

    base_to_file: Dict[str, Path] = {}
    for pdf_file in all_pdf_files:
        base_name = get_base_name(pdf_file.name)
        current_best = base_to_file.get(base_name)
        if current_best is None:
            base_to_file[base_name] = pdf_file
            continue

        if has_duplicate_suffix(current_best.name) and not has_duplicate_suffix(pdf_file.name):
            base_to_file[base_name] = pdf_file

    return list(base_to_file.values())


def build_manifest(pdf_folder: Path, unique_pdf_files: Iterable[Path]) -> dict:
    unique_pdf_files = list(unique_pdf_files)
    return {
        "pdf_folder": str(pdf_folder),
        "total_pdf_files": len(
            [path for path in pdf_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"]
        ),
        "unique_pdf_files": len(unique_pdf_files),
        "duplicate_pdf_files_removed": len(
            [path for path in pdf_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"]
        )
        - len(unique_pdf_files),
        "selected_filenames": [path.name for path in unique_pdf_files],
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def append_jsonl_record(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def materialize_checkpoint(checkpoint_jsonl: Path) -> tuple[Dict[str, str], Dict[str, dict]]:
    results: Dict[str, str] = {}
    failures: Dict[str, dict] = {}

    if not checkpoint_jsonl.exists():
        return results, failures

    with checkpoint_jsonl.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            doi = str(record["doi"])
            status = record.get("status")

            if status == "ok":
                results[doi] = record.get("text") or ""
                failures.pop(doi, None)
            else:
                results.pop(doi, None)
                failures[doi] = {
                    "filename": record.get("filename"),
                    "error": record.get("error"),
                    "page_count": record.get("page_count"),
                    "file_size_bytes": record.get("file_size_bytes"),
                    "completed_at": record.get("completed_at"),
                }

    return results, failures


def load_resume_state(checkpoint_jsonl: Path) -> tuple[set[str], Dict[str, dict]]:
    successful_dois: set[str] = set()
    failures: Dict[str, dict] = {}

    if not checkpoint_jsonl.exists():
        return successful_dois, failures

    with checkpoint_jsonl.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            doi = str(record["doi"])
            status = record.get("status")

            if status == "ok":
                successful_dois.add(doi)
                failures.pop(doi, None)
            else:
                successful_dois.discard(doi)
                failures[doi] = {
                    "filename": record.get("filename"),
                    "error": record.get("error"),
                    "page_count": record.get("page_count"),
                    "file_size_bytes": record.get("file_size_bytes"),
                    "completed_at": record.get("completed_at"),
                }

    return successful_dois, failures


def parse_single_pdf(pdf_path: str | Path) -> PdfParseResult:
    pdf_path = Path(pdf_path)
    doi = filename_to_doi(pdf_path.name)

    try:
        file_size = pdf_path.stat().st_size
        if file_size == 0:
            raise ValueError("Cannot open empty file")

        document = fitz.open(pdf_path)
        try:
            page_texts = [page.get_text() for page in document]
        finally:
            document.close()

        return PdfParseResult(
            doi=doi,
            filename=pdf_path.name,
            text="".join(page_texts),
            error=None,
            page_count=len(page_texts),
            file_size_bytes=file_size,
        )
    except Exception as exc:  # pragma: no cover - exercised in benchmark runs
        return PdfParseResult(
            doi=doi,
            filename=pdf_path.name,
            text=None,
            error=str(exc),
            page_count=0,
            file_size_bytes=0,
        )


def _default_workers(parallelism: Parallelism) -> int:
    cpu_count = os.cpu_count() or 1
    if parallelism == "process":
        return max(1, min(cpu_count - 1, 8))
    if parallelism == "thread":
        return max(1, min(cpu_count, 8))
    return 1


def parse_pdf_folder(
    pdf_folder: str | Path,
    output_json: str | Path,
    *,
    failures_json: str | Path | None = None,
    checkpoint_jsonl: str | Path | None = None,
    manifest_json: str | Path | None = None,
    parallelism: Parallelism = "thread",
    workers: Optional[int] = None,
    save_every: Optional[int] = 200,
    resume: bool = True,
    log_every: int = 25,
) -> tuple[Dict[str, str], Dict[str, dict], ParseRunStats]:
    pdf_folder = Path(pdf_folder)
    output_json = Path(output_json)
    failures_json = Path(failures_json) if failures_json else None
    checkpoint_jsonl = (
        Path(checkpoint_jsonl)
        if checkpoint_jsonl
        else output_json.with_name(f"{output_json.stem}_checkpoint.jsonl")
    )
    manifest_json = Path(manifest_json) if manifest_json else None

    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder does not exist: {pdf_folder}")

    if not resume:
        for path in (output_json, failures_json, checkpoint_jsonl):
            if path and path.exists():
                path.unlink()

    total_pdf_files = len([path for path in pdf_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"])
    unique_pdf_files = list_unique_pdf_files(pdf_folder)
    duplicate_pdf_files_removed = total_pdf_files - len(unique_pdf_files)

    if manifest_json:
        write_json_atomic(manifest_json, build_manifest(pdf_folder, unique_pdf_files))

    successful_dois: set[str] = set()
    latest_failures: Dict[str, dict] = {}
    if resume:
        successful_dois, latest_failures = load_resume_state(checkpoint_jsonl)

    queued_pdf_files = []
    for pdf_file in unique_pdf_files:
        doi = filename_to_doi(pdf_file.name)
        if doi in successful_dois:
            continue
        queued_pdf_files.append(pdf_file)

    skipped_existing = len(unique_pdf_files) - len(queued_pdf_files)
    workers = max(1, workers or _default_workers(parallelism))
    if parallelism == "none":
        workers = 1

    parsed_new = 0
    failed_new = 0
    checkpoint_write_count = 0
    flush_count = 0
    flush_elapsed_sec = 0.0
    total_text_chars_new = 0
    total_pages_new = 0
    completed = 0

    checkpoint_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def write_snapshot(force: bool = False) -> None:
        nonlocal flush_count, flush_elapsed_sec
        if not force and save_every is None:
            return
        if not force and save_every and completed % save_every != 0:
            return

        snapshot_started = time.perf_counter()
        results_snapshot, failures_snapshot = materialize_checkpoint(checkpoint_jsonl)
        write_json_atomic(output_json, results_snapshot)
        if failures_json:
            write_json_atomic(failures_json, failures_snapshot)
        flush_elapsed_sec += time.perf_counter() - snapshot_started
        flush_count += 1

    def handle_result(result: PdfParseResult, checkpoint_handle) -> None:
        nonlocal parsed_new, failed_new, total_text_chars_new, total_pages_new, completed, checkpoint_write_count
        completed += 1

        record = {
            "doi": result.doi,
            "filename": result.filename,
            "status": "error" if result.error else "ok",
            "page_count": result.page_count,
            "file_size_bytes": result.file_size_bytes,
            "completed_at": time.time(),
        }

        if result.error:
            record["error"] = result.error
            latest_failures[result.doi] = {
                "filename": result.filename,
                "error": result.error,
                "page_count": result.page_count,
                "file_size_bytes": result.file_size_bytes,
            }
            successful_dois.discard(result.doi)
            failed_new += 1
        else:
            text = result.text or ""
            record["text"] = text
            successful_dois.add(result.doi)
            latest_failures.pop(result.doi, None)
            parsed_new += 1
            total_text_chars_new += len(text)
            total_pages_new += result.page_count

        append_jsonl_record(checkpoint_handle, record)
        checkpoint_write_count += 1

        if log_every and completed % log_every == 0:
            print(
                f"Completed {completed}/{len(queued_pdf_files)} PDFs "
                f"(parsed={parsed_new}, failed={failed_new})"
            )

        write_snapshot(force=False)

    run_started = time.perf_counter()
    with checkpoint_jsonl.open("a", encoding="utf-8") as checkpoint_handle:
        if parallelism == "none" or len(queued_pdf_files) <= 1:
            for pdf_file in queued_pdf_files:
                handle_result(parse_single_pdf(pdf_file), checkpoint_handle)
        else:
            executor_cls = ThreadPoolExecutor if parallelism == "thread" else ProcessPoolExecutor
            with executor_cls(max_workers=workers) as executor:
                futures = {executor.submit(parse_single_pdf, pdf_file): pdf_file for pdf_file in queued_pdf_files}
                for future in as_completed(futures):
                    handle_result(future.result(), checkpoint_handle)

    write_snapshot(force=True)
    elapsed_sec = time.perf_counter() - run_started

    results, failures = materialize_checkpoint(checkpoint_jsonl)
    stats = ParseRunStats(
        pdf_folder=str(pdf_folder),
        output_json=str(output_json),
        failures_json=str(failures_json) if failures_json else None,
        checkpoint_jsonl=str(checkpoint_jsonl),
        manifest_json=str(manifest_json) if manifest_json else None,
        parallelism=parallelism,
        workers=workers,
        total_pdf_files=total_pdf_files,
        unique_pdf_files=len(unique_pdf_files),
        duplicate_pdf_files_removed=duplicate_pdf_files_removed,
        queued_pdf_files=len(queued_pdf_files),
        skipped_existing=skipped_existing,
        parsed_new=parsed_new,
        failed_new=failed_new,
        checkpoint_write_count=checkpoint_write_count,
        flush_count=flush_count,
        flush_elapsed_sec=round(flush_elapsed_sec, 3),
        total_text_chars_new=total_text_chars_new,
        total_pages_new=total_pages_new,
        elapsed_sec=round(elapsed_sec, 3),
    )
    return results, failures, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_folder", help="Folder containing PDF files")
    parser.add_argument("output_json", help="Path for the combined DOI-to-text JSON")
    parser.add_argument("--failures-json", default=None, help="Optional path for the failures JSON")
    parser.add_argument(
        "--checkpoint-jsonl",
        default=None,
        help="Optional append-only checkpoint file used for resume support",
    )
    parser.add_argument("--manifest-json", default=None, help="Optional path for the selected-file manifest JSON")
    parser.add_argument(
        "--parallelism",
        choices=("none", "thread", "process"),
        default="thread",
        help="How to parallelize PDF parsing",
    )
    parser.add_argument("--workers", type=int, default=None, help="Worker count for thread/process execution")
    parser.add_argument(
        "--save-every",
        type=int,
        default=200,
        help="How often to rebuild the combined JSON snapshot from the append-only checkpoint",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint/output files")
    parser.add_argument("--log-every", type=int, default=25, help="Print progress after this many completed PDFs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, stats = parse_pdf_folder(
        args.pdf_folder,
        args.output_json,
        failures_json=args.failures_json,
        checkpoint_jsonl=args.checkpoint_jsonl,
        manifest_json=args.manifest_json,
        parallelism=args.parallelism,
        workers=args.workers,
        save_every=args.save_every,
        resume=not args.no_resume,
        log_every=args.log_every,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
