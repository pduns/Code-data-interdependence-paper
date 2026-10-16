import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


_run_dir = os.environ.get("BATCH_RUN_DIR")
if not _run_dir:
    raise RuntimeError("BATCH_RUN_DIR must point to the batch run that should be retrieved.")
RUN_DIR = Path(_run_dir).expanduser().resolve()
BATCH_INFO_FILE = RUN_DIR / "batch_info.json"


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return api_key


def openai_get(api_key: str, url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def retrieve_batch(api_key: str, batch_id: str) -> dict:
    return json.loads(openai_get(api_key, f"https://api.openai.com/v1/batches/{batch_id}").decode("utf-8"))


def download_file(api_key: str, file_id: str, target_path: Path) -> None:
    target_path.write_bytes(openai_get(api_key, f"https://api.openai.com/v1/files/{file_id}/content"))


def extract_output_text(body):
    if body is None:
        return None
    output_text = body.get("output_text")
    if output_text:
        return output_text
    chunks = []
    for item in body.get("output", []):
        for content_item in item.get("content", []):
            if content_item.get("type") in {"output_text", "text"}:
                text_value = content_item.get("text")
                if isinstance(text_value, dict):
                    text_value = text_value.get("value")
                if text_value:
                    chunks.append(text_value)
    return "".join(chunks) if chunks else None


def normalize_label_value(value, strip_chars=";,."):
    if value is None:
        return ""
    value = re.sub(r"\s+", " ", str(value)).strip().strip(strip_chars)
    if not value:
        return ""
    if value.lower() == "manual":
        return "MANUAL"
    return value


def deduplicate_sources(sources):
    cleaned = []
    seen = set()
    for value in sources or []:
        source = normalize_label_value(value)
        if not source:
            continue
        key = source.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(source)
    return cleaned


def serialize_json_field(value):
    return json.dumps(value, ensure_ascii=False)


def parse_outputs(info: dict) -> pd.DataFrame:
    manifest_file = Path(info["manifest_file"])
    output_file = Path(info["batch_output_file"])
    error_file = Path(info["batch_error_file"])
    results_file = Path(info["results_file"])

    manifest_records = pd.read_csv(manifest_file).to_dict(orient="records")
    manifest_by_id = {row["custom_id"]: row for row in manifest_records}

    def make_base_row(base):
        return {
            "custom_id": base.get("custom_id"),
            "global_index": base.get("global_index"),
            "file_name": base.get("file_name"),
            "relative_path": base.get("relative_path"),
            "source_subfolder": base.get("source_subfolder"),
            "file_path": base.get("file_path"),
            "original_text_char_len": base.get("original_text_char_len"),
            "text_char_len": base.get("text_char_len"),
            "was_truncated": base.get("was_truncated"),
            "input_tokens_estimate": base.get("input_tokens_estimate"),
            "prompt_variant": info.get("prompt_variant"),
            "prompt_revision": info.get("prompt_revision"),
        }

    rows = []
    handled_ids = set()

    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = record.get("custom_id")
                handled_ids.add(custom_id)
                base = manifest_by_id.get(custom_id, {})
                response = record.get("response") or {}
                status_code = response.get("status_code")
                body = response.get("body") or {}
                output_text = extract_output_text(body)
                usage = body.get("usage") or {}

                row = make_base_row(base)
                if status_code != 200 or not output_text:
                    row.update({
                        "db_use": None,
                        "data_sources_json": None,
                        "data_resources_json": None,
                        "n_data_sources": None,
                        "reason_short": None,
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "response_status_code": status_code,
                        "status": f"response_error_{status_code}",
                        "error_message": "Missing output_text or non-200 response.",
                        "raw_output_text": output_text,
                    })
                    rows.append(row)
                    continue

                try:
                    parsed = json.loads(output_text)
                    data_sources = deduplicate_sources(parsed.get("data_sources", []))
                    db_use = int(parsed.get("db_use", 0))
                    if data_sources:
                        db_use = 1
                    row.update({
                        "db_use": db_use,
                        "data_sources_json": serialize_json_field(data_sources),
                        "data_resources_json": serialize_json_field(data_sources),
                        "n_data_sources": len(data_sources),
                        "reason_short": parsed.get("reason_short", ""),
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "response_status_code": status_code,
                        "status": "ok",
                        "error_message": None,
                        "raw_output_text": output_text,
                    })
                except Exception as exc:
                    row.update({
                        "db_use": None,
                        "data_sources_json": None,
                        "data_resources_json": None,
                        "n_data_sources": None,
                        "reason_short": None,
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "response_status_code": status_code,
                        "status": "parse_error",
                        "error_message": str(exc),
                        "raw_output_text": output_text,
                    })
                rows.append(row)

    if error_file.exists():
        with error_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = record.get("custom_id")
                handled_ids.add(custom_id)
                base = manifest_by_id.get(custom_id, {})
                error = record.get("error") or {}
                row = make_base_row(base)
                row.update({
                    "db_use": None,
                    "data_sources_json": None,
                    "data_resources_json": None,
                    "n_data_sources": None,
                    "reason_short": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "response_status_code": None,
                    "status": f"batch_error: {error.get('code', 'unknown')}",
                    "error_message": error.get("message", json.dumps(error)),
                    "raw_output_text": None,
                })
                rows.append(row)

    for custom_id, base in manifest_by_id.items():
        if custom_id in handled_ids:
            continue
        row = make_base_row(base)
        row.update({
            "db_use": None,
            "data_sources_json": None,
            "data_resources_json": None,
            "n_data_sources": None,
            "reason_short": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "response_status_code": None,
            "status": "missing_from_batch_output",
            "error_message": "No success or error line found for this custom_id.",
            "raw_output_text": None,
        })
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("custom_id").reset_index(drop=True)
    results_df.to_csv(results_file, index=False, encoding="utf-8-sig")
    return results_df


def main():
    api_key = require_api_key()
    info = json.loads(BATCH_INFO_FILE.read_text(encoding="utf-8"))
    batch = retrieve_batch(api_key, info["batch_id"])

    info.update({
        "status": batch.get("status"),
        "output_file_id": batch.get("output_file_id"),
        "error_file_id": batch.get("error_file_id"),
        "request_counts": batch.get("request_counts"),
        "completed_at": batch.get("completed_at"),
        "failed_at": batch.get("failed_at"),
        "expired_at": batch.get("expired_at"),
        "last_checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_api_raw_status": batch,
    })

    if batch.get("output_file_id"):
        output_path = Path(info["batch_output_file"])
        if not output_path.exists():
            download_file(api_key, batch["output_file_id"], output_path)
            print(f"Downloaded output: {output_path}")

    if batch.get("error_file_id"):
        error_path = Path(info["batch_error_file"])
        if not error_path.exists():
            download_file(api_key, batch["error_file_id"], error_path)
            print(f"Downloaded errors: {error_path}")

    BATCH_INFO_FILE.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "batch_id": info["batch_id"],
        "status": info.get("status"),
        "request_counts": info.get("request_counts"),
        "output_file_id": info.get("output_file_id"),
        "error_file_id": info.get("error_file_id"),
    }, indent=2))

    if Path(info["batch_output_file"]).exists() or Path(info["batch_error_file"]).exists():
        results_df = parse_outputs(info)
        n_ok = int((results_df["status"] == "ok").sum())
        n_err = int((results_df["status"] != "ok").sum())
        n_pos = int((results_df["db_use"] == 1).sum())
        input_tokens = int(results_df["input_tokens"].fillna(0).sum())
        output_tokens = int(results_df["output_tokens"].fillna(0).sum())
        print(f"Parsed results: {info['results_file']}")
        print(f"ok={n_ok} errors={n_err} positives={n_pos}")
        print(f"tokens input={input_tokens} output={output_tokens}")


if __name__ == "__main__":
    main()
