from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_config import DATA_ROOT


WHITESPACE_BEFORE_NEWLINE_RE = re.compile(r"[ \t]+\n")
WHITESPACE_AFTER_NEWLINE_RE = re.compile(r"\n[ \t]+")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTIBLANK_RE = re.compile(r"\n{3,}")
HEADING_NUMBER_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z].*$")
HEADING_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")
LOWERCASE_START_RE = re.compile(r"^[a-z]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

STATUS_FIELDNAMES = [
    "relative_path",
    "filename",
    "success",
    "paragraphs_total",
    "paragraphs_kept",
    "output_chars",
    "error",
]


@dataclass
class DocumentRecord:
    relative_path: str
    filename: str
    input_path: Path
    output_path: Path
    paragraphs: list[str]
    keep_mask: list[bool] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter parsed text files with a ModernBERT binary classifier."
    )
    parser.add_argument("--year", type=int, help="Shortcut for publications_<year> folders.")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=DATA_ROOT,
        help="Workspace root containing training_data, parsed_texts, and filtered_texts.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Model directory. Defaults to training_data/modernbert_db_prefilter_binary_finetuned.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input folder with .txt files. If omitted, derived from --year.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder for filtered files. If omitted, derived from --year.",
    )
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to 16 on CPU and 64 on CUDA.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Tokenizer max_length. Defaults to 256 on CPU for speed and 512 on CUDA.",
    )
    parser.add_argument("--max-paragraph-chars", type=int, default=1200)
    parser.add_argument(
        "--files-per-chunk",
        type=int,
        default=16,
        help="Number of files parsed before global paragraph batching.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional cap for quick tests or partial runs.",
    )
    parser.add_argument(
        "--status-csv",
        type=Path,
        help="Optional custom status CSV path. Defaults to <output-dir>/reduction_status.csv.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        help="Optional torch CPU thread count override.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    workspace_dir = args.workspace_dir.resolve()
    model_dir = (args.model_dir or workspace_dir / "training_data" / "modernbert_db_prefilter_binary_finetuned").resolve()

    if args.input_dir and args.output_dir:
        input_dir = args.input_dir.resolve()
        output_dir = args.output_dir.resolve()
    elif args.year is not None:
        input_dir = (workspace_dir / "parsed_texts" / f"publications_{args.year}").resolve()
        output_dir = (workspace_dir / "filtered_texts" / f"publications_{args.year}").resolve()
    else:
        raise SystemExit("Provide either --year or both --input-dir and --output-dir.")

    status_csv = (args.status_csv or output_dir / "reduction_status.csv").resolve()
    return workspace_dir, model_dir, input_dir, output_dir, status_csv


def parse_success(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def iter_existing_successes(status_csv: Path) -> set[str]:
    if not status_csv.exists():
        return set()

    processed: set[str] = set()
    with status_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not parse_success(row.get("success")):
                continue
            relative_path = (row.get("relative_path") or row.get("filename") or "").strip()
            if relative_path:
                processed.add(relative_path.replace("\\", "/"))
    return processed


def append_status_rows(status_csv: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    status_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = status_csv.exists()
    encoding = "utf-8" if file_exists else "utf-8-sig"
    with status_csv.open("a", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def clean_text(text: str | None) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\ufeff", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_BEFORE_NEWLINE_RE.sub("\n", text)
    text = WHITESPACE_AFTER_NEWLINE_RE.sub("\n", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTIBLANK_RE.sub("\n\n", text)
    return text.strip()


def is_probable_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.endswith((".", "!", "?")):
        return False
    if HEADING_NUMBER_RE.match(stripped):
        return True

    words = HEADING_WORD_RE.findall(stripped)
    if not words:
        return False
    if stripped.isupper() and len(words) <= 12:
        return True

    capitalized_ratio = sum(word[0].isupper() for word in words) / len(words)
    return len(words) <= 10 and capitalized_ratio >= 0.8


def merge_wrapped_lines_into_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current = ""

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if current.strip():
                paragraphs.append(current.strip())
                current = ""
            continue

        if is_probable_heading(line):
            if current.strip():
                paragraphs.append(current.strip())
                current = ""
            paragraphs.append(line)
            continue

        if not current:
            current = line
            continue

        if current.endswith("-") and LOWERCASE_START_RE.match(line):
            current = current[:-1] + line
        else:
            current = f"{current} {line}"

    if current.strip():
        paragraphs.append(current.strip())
    return paragraphs


def split_long_paragraph(paragraph: str, max_chars: int | None) -> list[str]:
    if not max_chars or len(paragraph) <= max_chars:
        return [paragraph]

    sentences = SENTENCE_SPLIT_RE.split(paragraph)
    if len(sentences) <= 1:
        return [paragraph]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not current:
            current = sentence
            continue

        candidate = f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            current = sentence

    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_into_paragraphs(text: str, max_paragraph_chars: int) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    paragraphs = merge_wrapped_lines_into_paragraphs(cleaned)
    final_paragraphs: list[str] = []

    for paragraph in paragraphs:
        paragraph = clean_text(paragraph)
        if not paragraph:
            continue
        final_paragraphs.extend(split_long_paragraph(paragraph, max_paragraph_chars))

    return [paragraph for paragraph in final_paragraphs if paragraph.strip()]


def batched(items: list[tuple[int, int, int, str]], batch_size: int) -> Iterable[list[tuple[int, int, int, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def make_status_row(
    relative_path: str,
    filename: str,
    success: bool,
    paragraphs_total: int | None,
    paragraphs_kept: int | None,
    output_chars: int | None,
    error: str = "",
) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "filename": filename,
        "success": success,
        "paragraphs_total": paragraphs_total,
        "paragraphs_kept": paragraphs_kept,
        "output_chars": output_chars,
        "error": error,
    }


def load_documents_for_chunk(
    file_paths: list[Path],
    input_dir: Path,
    output_dir: Path,
    max_paragraph_chars: int,
) -> tuple[list[DocumentRecord], list[dict[str, object]]]:
    documents: list[DocumentRecord] = []
    results: list[dict[str, object]] = []

    for input_path in file_paths:
        relative_path = input_path.relative_to(input_dir).as_posix()
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            text = read_text_file(input_path)
            paragraphs = split_into_paragraphs(text, max_paragraph_chars=max_paragraph_chars)
        except Exception as exc:  # noqa: BLE001
            results.append(
                make_status_row(
                    relative_path=relative_path,
                    filename=input_path.name,
                    success=False,
                    paragraphs_total=None,
                    paragraphs_kept=None,
                    output_chars=None,
                    error=str(exc),
                )
            )
            continue

        if not paragraphs:
            try:
                output_path.write_text("", encoding="utf-8")
                results.append(
                    make_status_row(
                        relative_path=relative_path,
                        filename=input_path.name,
                        success=True,
                        paragraphs_total=0,
                        paragraphs_kept=0,
                        output_chars=0,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    make_status_row(
                        relative_path=relative_path,
                        filename=input_path.name,
                        success=False,
                        paragraphs_total=0,
                        paragraphs_kept=0,
                        output_chars=0,
                        error=str(exc),
                    )
                )
            continue

        documents.append(
            DocumentRecord(
                relative_path=relative_path,
                filename=input_path.name,
                input_path=input_path,
                output_path=output_path,
                paragraphs=paragraphs,
                keep_mask=[False] * len(paragraphs),
            )
        )

    return documents, results


@torch.inference_mode()
def predict_documents(
    documents: list[DocumentRecord],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int,
    max_length: int,
    threshold: float,
) -> tuple[int, int]:
    paragraph_refs: list[tuple[int, int, int, str]] = []
    for doc_index, document in enumerate(documents):
        for paragraph_index, paragraph in enumerate(document.paragraphs):
            paragraph_refs.append((len(paragraph), doc_index, paragraph_index, paragraph))

    paragraph_refs.sort(key=lambda item: item[0])
    batch_count = math.ceil(len(paragraph_refs) / batch_size)

    for batch in batched(paragraph_refs, batch_size):
        batch_texts = [item[3] for item in batch]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in encoded.items()
        }

        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=1)[:, 1].detach().cpu().tolist()

        for (_, doc_index, paragraph_index, _), probability in zip(batch, probabilities):
            documents[doc_index].keep_mask[paragraph_index] = probability >= threshold

    return len(paragraph_refs), batch_count


def finalize_documents(documents: list[DocumentRecord]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    for document in documents:
        kept_paragraphs = [
            paragraph
            for paragraph, keep in zip(document.paragraphs, document.keep_mask)
            if keep
        ]
        reduced_text = "\n\n".join(kept_paragraphs)

        try:
            document.output_path.write_text(reduced_text, encoding="utf-8")
            results.append(
                make_status_row(
                    relative_path=document.relative_path,
                    filename=document.filename,
                    success=True,
                    paragraphs_total=len(document.paragraphs),
                    paragraphs_kept=len(kept_paragraphs),
                    output_chars=len(reduced_text),
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                make_status_row(
                    relative_path=document.relative_path,
                    filename=document.filename,
                    success=False,
                    paragraphs_total=len(document.paragraphs),
                    paragraphs_kept=len(kept_paragraphs),
                    output_chars=len(reduced_text),
                    error=str(exc),
                )
            )

    return results


def chunked_paths(paths: list[Path], chunk_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(paths), chunk_size):
        yield paths[start : start + chunk_size]


def main() -> None:
    args = parse_args()
    _, model_dir, input_dir, output_dir, status_csv = resolve_paths(args)

    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or (64 if device.type == "cuda" else 16)
    max_length = args.max_length or (512 if device.type == "cuda" else 256)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Status CSV: {status_csv}")
    print(f"Model directory: {model_dir}")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    print(f"Files per chunk: {args.files_per_chunk}")
    print(f"Torch threads: {torch.get_num_threads()}")
    if device.type == "cpu" and args.max_length is None:
        print("CPU mode: using max_length=256 for faster filtering. Use --max-length 512 for notebook-like settings.")

    all_files = sorted(path for path in input_dir.rglob("*.txt") if path.is_file())
    processed_files = iter_existing_successes(status_csv)

    files_to_process = [
        path
        for path in all_files
        if path.relative_to(input_dir).as_posix() not in processed_files
    ]

    if args.max_files is not None:
        files_to_process = files_to_process[: args.max_files]

    print(f"Found {len(all_files)} text files.")
    print(f"Already successfully processed: {len(processed_files)}")
    print(f"Remaining files to process: {len(files_to_process)}")

    if not files_to_process:
        print("Nothing to do.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    total_start = time.perf_counter()
    successful_new = 0

    for chunk_index, file_chunk in enumerate(
        chunked_paths(files_to_process, args.files_per_chunk),
        start=1,
    ):
        chunk_start = time.perf_counter()
        docs, chunk_rows = load_documents_for_chunk(
            file_paths=file_chunk,
            input_dir=input_dir,
            output_dir=output_dir,
            max_paragraph_chars=args.max_paragraph_chars,
        )

        paragraph_count = 0
        batch_count = 0
        if docs:
            try:
                paragraph_count, batch_count = predict_documents(
                    documents=docs,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                    threshold=args.threshold,
                )
                chunk_rows.extend(finalize_documents(docs))
            except Exception as exc:  # noqa: BLE001
                for document in docs:
                    chunk_rows.append(
                        make_status_row(
                            relative_path=document.relative_path,
                            filename=document.filename,
                            success=False,
                            paragraphs_total=len(document.paragraphs),
                            paragraphs_kept=None,
                            output_chars=None,
                            error=str(exc),
                        )
                    )

        append_status_rows(status_csv, chunk_rows)
        successful_new += sum(1 for row in chunk_rows if parse_success(row["success"]))

        processed_now = min(chunk_index * args.files_per_chunk, len(files_to_process))
        chunk_seconds = time.perf_counter() - chunk_start
        elapsed_total = time.perf_counter() - total_start
        avg_file_seconds = elapsed_total / max(processed_now, 1)
        remaining_files = len(files_to_process) - processed_now
        eta_minutes = (remaining_files * avg_file_seconds) / 60

        print(
            f"[chunk {chunk_index}] files={len(file_chunk)} "
            f"paragraphs={paragraph_count} batches={batch_count} "
            f"time={chunk_seconds:.2f}s elapsed={elapsed_total/60:.1f}m "
            f"avg/file={avg_file_seconds:.2f}s ETA={eta_minutes:.1f}m"
        )

    total_seconds = time.perf_counter() - total_start
    print("Done.")
    print(f"Processed in this run: {len(files_to_process)} files")
    print(f"Successful in this run: {successful_new}")
    print(f"Total runtime: {total_seconds/60:.2f} minutes")
    print(f"Average per file: {total_seconds / max(len(files_to_process), 1):.2f} seconds")


if __name__ == "__main__":
    main()
