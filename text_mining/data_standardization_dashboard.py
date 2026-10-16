"""
Local dashboard for standardizing LLM-detected data source names.

Run from the project root with:
    python code/data_standardization_dashboard.py

The app intentionally uses only the Python standard library so it can run in
this workspace without installing dashboard dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import argparse
import sys
import tempfile
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_config import DATA_ROOT


PROJECT_ROOT = DATA_ROOT
INPUT_ROWS_PATH = DATA_ROOT / "llm_labelled_data" / "compiled" / "llm_labelled_data_sources_long.csv"
PROCESSED_DIR = DATA_ROOT / "processed_data"
DICTIONARY_PATH = PROCESSED_DIR / "data_holder_dictionary.csv"
DECISIONS_PATH = PROCESSED_DIR / "data_source_standardization_decisions.csv"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

HOLDER_FIELDS = [
    "holder_id",
    "canonical_name",
    "aliases",
    "homepage_url",
    "notes",
    "created_at_utc",
    "updated_at_utc",
]

DECISION_FIELDS = [
    "review_id",
    "global_index",
    "relative_path",
    "source_subfolder",
    "file_name",
    "source_order",
    "raw_data_source",
    "llm_reason_short",
    "decision_type",
    "standardized_resource_name",
    "canonical_holder_id",
    "canonical_holder_name",
    "evidence_basis",
    "evidence_note",
    "coder_note",
    "updated_at_utc",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: str | None) -> str:
    value = value or ""
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def make_review_id(row: dict[str, str]) -> str:
    basis = "|".join(
        [
            row.get("global_index", ""),
            row.get("relative_path", ""),
            row.get("source_order", ""),
            row.get("data_source", ""),
        ]
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        temp_path = Path(handle.name)
    temp_path.replace(path)


def ensure_storage_files() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not DICTIONARY_PATH.exists():
        atomic_write_csv(DICTIONARY_PATH, HOLDER_FIELDS, [])
    if not DECISIONS_PATH.exists():
        atomic_write_csv(DECISIONS_PATH, DECISION_FIELDS, [])


def load_review_rows() -> list[dict[str, str]]:
    if not INPUT_ROWS_PATH.exists():
        raise FileNotFoundError(f"Missing input rows: {INPUT_ROWS_PATH}")

    rows = read_csv(INPUT_ROWS_PATH)
    review_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        raw_source = (row.get("data_source") or "").strip()
        if not raw_source:
            continue
        review_id = make_review_id(row)
        if review_id in seen:
            continue
        seen.add(review_id)
        review_rows.append(
            {
                "review_id": review_id,
                "global_index": row.get("global_index", ""),
                "relative_path": row.get("relative_path", ""),
                "source_subfolder": row.get("source_subfolder", ""),
                "file_name": row.get("file_name", ""),
                "db_use": row.get("db_use", ""),
                "data_source": raw_source,
                "source_order": row.get("source_order", ""),
                "reason_short": row.get("reason_short", ""),
                "run_type": row.get("run_type", ""),
                "run_name": row.get("run_name", ""),
                "prompt_revision": row.get("prompt_revision", ""),
                "status": row.get("status", ""),
                "source_norm": normalize_text(raw_source),
            }
        )
    return review_rows


def load_decisions_by_id() -> dict[str, dict[str, str]]:
    return {row.get("review_id", ""): row for row in read_csv(DECISIONS_PATH) if row.get("review_id")}


def load_holders() -> list[dict[str, str]]:
    holders = read_csv(DICTIONARY_PATH)
    return sorted(holders, key=lambda row: normalize_text(row.get("canonical_name", "")))


def next_holder_id(holders: list[dict[str, str]]) -> str:
    max_n = 0
    for holder in holders:
        match = re.search(r"(\d+)$", holder.get("holder_id", ""))
        if match:
            max_n = max(max_n, int(match.group(1)))
    return f"DH{max_n + 1:06d}"


def upsert_decision(payload: dict[str, Any]) -> dict[str, str]:
    rows = read_csv(DECISIONS_PATH)
    by_id = {row.get("review_id", ""): row for row in rows if row.get("review_id")}
    review_id = str(payload.get("review_id", "")).strip()
    if not review_id:
        raise ValueError("review_id is required")

    clean_row = {field: str(payload.get(field, "") or "").strip() for field in DECISION_FIELDS}
    clean_row["review_id"] = review_id
    clean_row["updated_at_utc"] = utc_now()
    by_id[review_id] = clean_row

    ordered: list[dict[str, str]] = []
    seen = set()
    for row in rows:
        existing_id = row.get("review_id", "")
        if existing_id == review_id:
            ordered.append(clean_row)
            seen.add(review_id)
        elif existing_id and existing_id not in seen:
            ordered.append(by_id[existing_id])
            seen.add(existing_id)
    if review_id not in seen:
        ordered.append(clean_row)
    atomic_write_csv(DECISIONS_PATH, DECISION_FIELDS, ordered)
    return clean_row


def add_holder(payload: dict[str, Any]) -> dict[str, str]:
    canonical_name = str(payload.get("canonical_name", "") or "").strip()
    if not canonical_name:
        raise ValueError("canonical_name is required")

    holders = read_csv(DICTIONARY_PATH)
    now = utc_now()
    canonical_norm = normalize_text(canonical_name)
    for holder in holders:
        if normalize_text(holder.get("canonical_name", "")) == canonical_norm:
            return holder

    holder = {
        "holder_id": next_holder_id(holders),
        "canonical_name": canonical_name,
        "aliases": str(payload.get("aliases", "") or "").strip(),
        "homepage_url": str(payload.get("homepage_url", "") or "").strip(),
        "notes": str(payload.get("notes", "") or "").strip(),
        "created_at_utc": now,
        "updated_at_utc": now,
    }
    holders.append(holder)
    atomic_write_csv(DICTIONARY_PATH, HOLDER_FIELDS, holders)
    return holder


def read_source_text(relative_path: str) -> dict[str, str]:
    if not relative_path:
        return {"text": "", "path": "", "warning": "No relative path is available for this row."}
    source_path = (PROJECT_ROOT / "filtered_texts" / relative_path).resolve()
    filtered_root = (PROJECT_ROOT / "filtered_texts").resolve()
    if filtered_root not in source_path.parents:
        return {"text": "", "path": "", "warning": "Requested text path is outside filtered_texts."}
    if not source_path.exists():
        return {"text": "", "path": str(source_path), "warning": "Source text file was not found."}
    text = source_path.read_text(encoding="utf-8", errors="replace")
    if len(text) > 30000:
        return {
            "text": text[:30000],
            "path": str(source_path),
            "warning": "Showing the first 30,000 characters only.",
        }
    return {"text": text, "path": str(source_path), "warning": ""}


def bootstrap_payload() -> dict[str, Any]:
    rows = load_review_rows()
    decisions = load_decisions_by_id()
    holders = load_holders()
    done = sum(1 for row in rows if row["review_id"] in decisions)
    return {
        "rows": rows,
        "decisions": decisions,
        "holders": holders,
        "counts": {
            "rows": len(rows),
            "done": done,
            "remaining": len(rows) - done,
            "holders": len(holders),
        },
        "paths": {
            "input_rows": str(INPUT_ROWS_PATH),
            "dictionary": str(DICTIONARY_PATH),
            "decisions": str(DECISIONS_PATH),
        },
    }


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Data Source Standardization</title>
  <style>
    :root {
      --bg: #f7f5ef;
      --card: #ffffff;
      --ink: #1f2933;
      --muted: #6b7280;
      --line: #d9ded7;
      --accent: #256f63;
      --accent-soft: #e5f2ef;
      --warn-soft: #fff4d8;
      --radius: 14px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Aptos", sans-serif;
      font-size: 15px;
    }

    button, input, select, textarea { font: inherit; }

    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px;
    }

    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 14px;
    }

    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
    }

    .subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }

    .metrics {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .metric {
      min-width: 84px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--card);
      text-align: right;
    }

    .metric strong {
      display: block;
      font-size: 20px;
      line-height: 1;
    }

    .metric span {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .layout {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--card);
    }

    .queue {
      position: sticky;
      top: 14px;
      max-height: calc(100vh - 28px);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .panel-header {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }

    .panel-header h2 {
      margin: 0;
      font-size: 16px;
    }

    .hint, .meta, .small {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }

    .hint { margin: 5px 0 0; }

    .controls {
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }

    .input, .select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      outline: none;
    }

    textarea {
      min-height: 94px;
      resize: vertical;
      line-height: 1.4;
    }

    .input:focus, .select:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(37, 111, 99, 0.12);
    }

    .queue-list {
      overflow: auto;
      padding: 8px;
    }

    .queue-card {
      width: 100%;
      display: block;
      margin: 0 0 7px;
      padding: 10px;
      border: 1px solid transparent;
      border-radius: 11px;
      background: #f9faf7;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
    }

    .queue-card:hover { border-color: var(--line); }

    .queue-card.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }

    .queue-card.done .source-title:after {
      content: "saved";
      margin-left: 6px;
      color: var(--accent);
      font-size: 11px;
      font-weight: 700;
    }

    .source-title {
      font-weight: 700;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }

    .review {
      padding: 16px;
    }

    .summary {
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fbfcfa;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }

    .label-title {
      margin: 0;
      font-size: 30px;
      line-height: 1.12;
      overflow-wrap: anywhere;
    }

    .reason {
      margin-top: 10px;
      padding: 12px;
      border-left: 4px solid var(--accent);
      background: #fff;
      line-height: 1.5;
    }

    .file-name {
      display: inline-block;
      margin-top: 10px;
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: #fff;
      font-size: 12px;
    }

    .label-list {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-top: 8px;
    }

    .label-chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: #fff;
      color: var(--ink);
      font-size: 13px;
    }

    .label-chip.current {
      border-color: var(--accent);
      background: var(--accent-soft);
      font-weight: 700;
    }

    .section {
      margin-top: 12px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fff;
    }

    .section h3 {
      margin: 0 0 10px;
      font-size: 16px;
    }

    .decision-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
    }

    .choice {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 48px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f9faf7;
      text-align: center;
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
      line-height: 1.2;
    }

    .choice input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }

    .choice:has(input:checked) {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: #134e46;
    }

    .selected-holder {
      margin-top: 9px;
      padding: 10px;
      border: 1px solid var(--accent);
      border-radius: 10px;
      background: var(--accent-soft);
      font-weight: 700;
    }

    .empty-selected {
      border-color: #e4c66a;
      background: var(--warn-soft);
      color: #7c5a00;
      font-weight: 600;
    }

    .holder-suggestions {
      display: grid;
      gap: 7px;
      margin-top: 8px;
      max-height: 230px;
      overflow: auto;
    }

    .holder-card {
      width: 100%;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f9faf7;
      text-align: left;
      cursor: pointer;
    }

    .holder-card:hover, .holder-card.selected {
      border-color: var(--accent);
      background: var(--accent-soft);
    }

    .holder-name { font-weight: 700; }

    .add-holder-form {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }

    .add-holder-form .full { grid-column: 1 / -1; }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }

    .button {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 9px 12px;
      background: #fff;
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
    }

    .button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }

    .toast {
      position: fixed;
      right: 18px;
      bottom: 18px;
      max-width: 360px;
      padding: 11px 13px;
      border-radius: 10px;
      background: #1f2933;
      color: #fff;
      opacity: 0;
      transform: translateY(8px);
      transition: 160ms ease;
      pointer-events: none;
    }

    .toast.visible {
      opacity: 1;
      transform: translateY(0);
    }

    .hidden { display: none !important; }

    @media (max-width: 900px) {
      header, .layout, .summary-grid {
        grid-template-columns: 1fr;
        display: grid;
      }

      header { align-items: start; }

      .queue {
        position: static;
        max-height: none;
      }

      .queue-list { max-height: 360px; }

      .decision-grid, .add-holder-form {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>Data source standardization</h1>
        <p class="subtitle">Review each LLM label, map it to a canonical data holder, or mark it for exclusion/review.</p>
      </div>
      <div class="metrics">
        <div class="metric"><strong id="metricRows">0</strong><span>Rows</span></div>
        <div class="metric"><strong id="metricDone">0</strong><span>Saved</span></div>
        <div class="metric"><strong id="metricRemaining">0</strong><span>Left</span></div>
        <div class="metric"><strong id="metricHolders">0</strong><span>Holders</span></div>
      </div>
    </header>

    <section class="layout">
      <aside class="panel queue">
        <div class="panel-header">
          <h2>Queue</h2>
          <p id="pathHint" class="hint">Loading data...</p>
          <div class="controls">
            <input id="queueSearch" class="input" placeholder="Search label, reasoning, file...">
            <select id="queueFilter" class="select">
              <option value="todo">Unreviewed first</option>
              <option value="all">All rows</option>
              <option value="done">Saved only</option>
              <option value="needs_review">Needs review</option>
              <option value="false_positive">False positives</option>
              <option value="manual_unresolved">Manual / unresolved</option>
            </select>
          </div>
        </div>
        <div id="queueList" class="queue-list"></div>
      </aside>

      <section class="panel review">
        <div id="emptyState" class="section">
          <h3>No row selected yet</h3>
          <p class="hint">The dashboard is loading, or no review rows were found.</p>
        </div>

        <div id="reviewPane" class="hidden">
          <div class="summary">
            <h2 id="sourceName" class="label-title"></h2>
            <div id="reasonShort" class="reason"></div>
            <div id="fileName" class="file-name"></div>
            <div class="summary-grid">
              <div>
                <div class="small"><strong>Labels returned for this document</strong></div>
                <div id="documentLabels" class="label-list"></div>
              </div>
              <div>
                <div class="small"><strong>Document</strong></div>
                <div id="documentMeta" class="meta"></div>
              </div>
            </div>
          </div>

          <section class="section">
            <h3>1. What is this encountered label?</h3>
            <div class="decision-grid">
              <label class="choice"><input type="radio" name="decisionType" value="data_resource">Data resource</label>
              <label class="choice"><input type="radio" name="decisionType" value="data_holder">Data holder</label>
              <label class="choice"><input type="radio" name="decisionType" value="false_positive">False positive</label>
              <label class="choice"><input type="radio" name="decisionType" value="manual_unresolved">Manual / unresolved</label>
              <label class="choice"><input type="radio" name="decisionType" value="needs_review">Needs review</label>
            </div>
          </section>

          <section class="section">
            <h3>2. Select or add data holder</h3>
            <input id="holderSearch" class="input" placeholder="Type to search existing holders...">
            <div id="selectedHolder" class="selected-holder empty-selected">No holder selected yet</div>
            <div id="holderSuggestions" class="holder-suggestions"></div>

            <form id="addHolderForm" class="add-holder-form">
              <input id="newHolderName" class="input full" placeholder="Add new data holder name" required>
              <input id="newHolderUrl" class="input" placeholder="Homepage or evidence URL">
              <textarea id="newHolderNotes" class="full" placeholder="Optional holder note"></textarea>
              <button class="button primary full" type="submit">Add holder and select</button>
            </form>
          </section>

          <section class="section">
            <h3>3. Note</h3>
            <textarea id="coderNote" placeholder="Optional note for uncertainty, source, or later adjudication."></textarea>
          </section>

          <div class="actions">
            <button id="saveNextButton" class="button primary" type="button">Save and next</button>
            <button id="saveButton" class="button" type="button">Save</button>
            <button id="clearHolderButton" class="button" type="button">Clear holder</button>
            <button id="nextButton" class="button" type="button">Skip to next</button>
          </div>
        </div>
      </section>
    </section>
  </main>
  <div id="toast" class="toast"></div>

  <script>
    const state = {
      rows: [],
      decisions: {},
      holders: [],
      currentReviewId: null,
      selectedHolderId: "",
      queueFilter: "todo",
      queueSearch: "",
      holderSearch: "",
    };

    const $ = (id) => document.getElementById(id);

    function normalize(value) {
      return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    }

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function toast(message) {
      const el = $("toast");
      el.textContent = message;
      el.classList.add("visible");
      clearTimeout(window.__toastTimer);
      window.__toastTimer = setTimeout(() => el.classList.remove("visible"), 2400);
    }

    async function apiJson(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
      return payload;
    }

    function currentRow() {
      return state.rows.find((row) => row.review_id === state.currentReviewId) || null;
    }

    function currentDecision() {
      return state.decisions[state.currentReviewId] || null;
    }

    function selectedHolder() {
      return state.holders.find((holder) => holder.holder_id === state.selectedHolderId) || null;
    }

    function isRowSaved(row) {
      return Boolean(state.decisions[row.review_id]);
    }

    function decisionTypeFor(row) {
      return (state.decisions[row.review_id] || {}).decision_type || "";
    }

    function documentRows(row) {
      if (!row) return [];
      return state.rows.filter((candidate) =>
        candidate.relative_path === row.relative_path ||
        (candidate.global_index && candidate.global_index === row.global_index)
      );
    }

    function filteredRows() {
      const needle = normalize(state.queueSearch);
      let rows = [...state.rows];
      if (needle) {
        rows = rows.filter((row) => normalize([
          row.data_source,
          row.reason_short,
          row.relative_path,
          row.file_name,
          row.global_index,
        ].join(" ")).includes(needle));
      }
      if (state.queueFilter === "done") {
        rows = rows.filter(isRowSaved);
      } else if (state.queueFilter === "todo") {
        rows.sort((a, b) => Number(isRowSaved(a)) - Number(isRowSaved(b)));
      } else if (state.queueFilter !== "all") {
        rows = rows.filter((row) => decisionTypeFor(row) === state.queueFilter);
      }
      return rows;
    }

    function renderMetrics() {
      const done = Object.keys(state.decisions).length;
      $("metricRows").textContent = state.rows.length;
      $("metricDone").textContent = done;
      $("metricRemaining").textContent = Math.max(0, state.rows.length - done);
      $("metricHolders").textContent = state.holders.length;
    }

    function renderQueue() {
      const rows = filteredRows();
      const list = $("queueList");
      if (!rows.length) {
        list.innerHTML = `<div class="hint" style="padding:10px;">No rows match this filter.</div>`;
        return;
      }
      list.innerHTML = rows.map((row) => {
        const active = row.review_id === state.currentReviewId ? " active" : "";
        const done = isRowSaved(row) ? " done" : "";
        return `
          <button class="queue-card${active}${done}" data-review-id="${escapeHtml(row.review_id)}">
            <div class="source-title">${escapeHtml(row.data_source)}</div>
            <div class="meta">#${escapeHtml(row.global_index)} | source ${escapeHtml(row.source_order)} | ${escapeHtml(row.file_name)}</div>
          </button>`;
      }).join("");
      list.querySelectorAll(".queue-card").forEach((button) => {
        button.addEventListener("click", () => selectRow(button.dataset.reviewId));
      });
    }

    function holderScore(holder, row) {
      const query = normalize(state.holderSearch);
      const source = normalize(row?.data_source || "");
      const name = normalize(holder.canonical_name);
      if (!name) return 0;
      if (query) {
        if (name === query) return 100;
        if (name.startsWith(query)) return 90;
        if (name.includes(query)) return 60;
        return 0;
      }
      if (!source) return 0;
      if (name === source) return 100;
      if (source.startsWith(name) || name.startsWith(source)) return 90;
      const firstToken = source.split(" ")[0] || "";
      if (firstToken.length >= 3 && name.startsWith(firstToken)) return 80;
      return 0;
    }

    function suggestedHolders() {
      const row = currentRow();
      return [...state.holders]
        .map((holder) => ({...holder, score: holderScore(holder, row)}))
        .filter((holder) => holder.score > 0)
        .sort((a, b) => (b.score - a.score) || normalize(a.canonical_name).localeCompare(normalize(b.canonical_name)))
        .slice(0, 30);
    }

    function renderHolderSuggestions() {
      const suggestions = suggestedHolders();
      const list = $("holderSuggestions");
      if (!state.holders.length) {
        list.innerHTML = `<div class="hint">No data holders exist yet. Add the first one below.</div>`;
        return;
      }
      if (!suggestions.length) {
        list.innerHTML = `<div class="hint">No prefix matches. Type more above or add a new holder below.</div>`;
        return;
      }
      list.innerHTML = suggestions.map((holder) => {
        const selected = holder.holder_id === state.selectedHolderId ? " selected" : "";
        return `
          <button class="holder-card${selected}" data-holder-id="${escapeHtml(holder.holder_id)}">
            <div class="holder-name">${escapeHtml(holder.canonical_name)}</div>
            <div class="meta">${escapeHtml(holder.holder_id)}${holder.notes ? " | " + escapeHtml(holder.notes) : ""}</div>
          </button>`;
      }).join("");
      list.querySelectorAll(".holder-card").forEach((button) => {
        button.addEventListener("click", () => selectHolder(button.dataset.holderId));
      });
    }

    function renderSelectedHolder() {
      const holder = selectedHolder();
      const el = $("selectedHolder");
      if (!holder) {
        el.classList.add("empty-selected");
        el.textContent = "No holder selected yet";
        return;
      }
      el.classList.remove("empty-selected");
      el.textContent = `${holder.canonical_name} (${holder.holder_id})`;
    }

    function selectHolder(holderId) {
      state.selectedHolderId = holderId || "";
      renderSelectedHolder();
      renderHolderSuggestions();
    }

    function setDecisionType(value) {
      document.querySelectorAll('input[name="decisionType"]').forEach((input) => {
        input.checked = input.value === value;
      });
    }

    function renderDocumentLabels(row) {
      const rows = documentRows(row);
      const seen = new Set();
      const labels = [];
      for (const item of rows) {
        const key = normalize(item.data_source);
        if (!key || seen.has(key)) continue;
        seen.add(key);
        labels.push(item);
      }
      $("documentLabels").innerHTML = labels.map((item) => {
        const current = item.review_id === row.review_id ? " current" : "";
        return `<span class="label-chip${current}">${escapeHtml(item.data_source)}</span>`;
      }).join("") || `<span class="hint">No other labels returned for this document.</span>`;
    }

    function selectRow(reviewId) {
      state.currentReviewId = reviewId;
      const row = currentRow();
      if (!row) return;
      const decision = currentDecision();
      state.selectedHolderId = decision?.canonical_holder_id || "";
      state.holderSearch = "";
      $("holderSearch").value = "";

      $("emptyState").classList.add("hidden");
      $("reviewPane").classList.remove("hidden");
      $("sourceName").textContent = row.data_source;
      $("reasonShort").textContent = row.reason_short || "No LLM reasoning was supplied for this row.";
      $("fileName").textContent = `file_name: ${row.file_name || ""}`;
      $("documentMeta").innerHTML = [
        ["global_index", row.global_index],
        ["source_order", row.source_order],
        ["folder", row.source_subfolder],
        ["prompt", row.prompt_revision],
      ].filter(([, value]) => value).map(([label, value]) => `${escapeHtml(label)}: ${escapeHtml(value)}`).join("<br>");
      renderDocumentLabels(row);
      $("coderNote").value = decision?.coder_note || decision?.evidence_note || "";
      setDecisionType(decision?.decision_type || "data_resource");
      renderSelectedHolder();
      renderQueue();
      renderHolderSuggestions();
    }

    function nextUnreviewedId(afterCurrent = true) {
      if (!state.rows.length) return null;
      const startIndex = Math.max(0, state.rows.findIndex((row) => row.review_id === state.currentReviewId));
      const ordered = afterCurrent
        ? [...state.rows.slice(startIndex + 1), ...state.rows.slice(0, startIndex + 1)]
        : state.rows;
      return (ordered.find((row) => !isRowSaved(row)) || ordered[0] || {}).review_id || null;
    }

    function collectDecisionPayload() {
      const row = currentRow();
      if (!row) throw new Error("No row selected");
      const decisionType = document.querySelector('input[name="decisionType"]:checked')?.value || "data_resource";
      const holder = selectedHolder();
      return {
        review_id: row.review_id,
        global_index: row.global_index,
        relative_path: row.relative_path,
        source_subfolder: row.source_subfolder,
        file_name: row.file_name,
        source_order: row.source_order,
        raw_data_source: row.data_source,
        llm_reason_short: row.reason_short,
        decision_type: decisionType,
        standardized_resource_name: "",
        canonical_holder_id: holder?.holder_id || "",
        canonical_holder_name: holder?.canonical_name || "",
        evidence_basis: "",
        evidence_note: "",
        coder_note: $("coderNote").value.trim(),
      };
    }

    async function saveDecision(moveNext = false) {
      try {
        const payload = collectDecisionPayload();
        const saved = await apiJson("/api/decision", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        state.decisions[saved.review_id] = saved;
        renderMetrics();
        renderQueue();
        toast("Decision saved.");
        if (moveNext) {
          const nextId = nextUnreviewedId(true);
          if (nextId) selectRow(nextId);
        }
      } catch (error) {
        toast(error.message);
      }
    }

    async function addHolder(event) {
      event.preventDefault();
      try {
        const payload = {
          canonical_name: $("newHolderName").value.trim(),
          aliases: "",
          homepage_url: $("newHolderUrl").value.trim(),
          notes: $("newHolderNotes").value.trim(),
        };
        const holder = await apiJson("/api/holder", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        const existingIndex = state.holders.findIndex((item) => item.holder_id === holder.holder_id);
        if (existingIndex >= 0) state.holders[existingIndex] = holder;
        else state.holders.push(holder);
        $("addHolderForm").reset();
        selectHolder(holder.holder_id);
        renderMetrics();
        toast("Holder added and selected.");
      } catch (error) {
        toast(error.message);
      }
    }

    async function init() {
      try {
        const payload = await apiJson("/api/bootstrap");
        state.rows = payload.rows || [];
        state.decisions = payload.decisions || {};
        state.holders = payload.holders || [];
        $("pathHint").textContent = `${state.rows.length} labels loaded`;
        renderMetrics();
        renderQueue();
        const firstId = nextUnreviewedId(false);
        if (firstId) selectRow(firstId);
      } catch (error) {
        $("emptyState").classList.remove("hidden");
        $("emptyState").innerHTML = `<h3>Dashboard could not load</h3><p class="hint">${escapeHtml(error.message)}</p>`;
        toast(error.message);
      }
    }

    $("queueSearch").addEventListener("input", (event) => {
      state.queueSearch = event.target.value;
      renderQueue();
    });
    $("queueFilter").addEventListener("change", (event) => {
      state.queueFilter = event.target.value;
      renderQueue();
    });
    $("holderSearch").addEventListener("input", (event) => {
      state.holderSearch = event.target.value;
      renderHolderSuggestions();
    });
    document.querySelectorAll('input[name="decisionType"]').forEach((input) => {
      input.addEventListener("change", () => setDecisionType(input.value));
    });
    $("saveButton").addEventListener("click", () => saveDecision(false));
    $("saveNextButton").addEventListener("click", () => saveDecision(true));
    $("nextButton").addEventListener("click", () => {
      const nextId = nextUnreviewedId(true);
      if (nextId) selectRow(nextId);
    });
    $("clearHolderButton").addEventListener("click", () => selectHolder(""));
    $("addHolderForm").addEventListener("submit", addHolder);

    init();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DataStandardizationDashboard/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        if not raw:
            return {}
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("JSON payload must be an object")
        return payload

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self.send_html(APP_HTML)
            elif parsed.path == "/api/bootstrap":
                self.send_json(bootstrap_payload())
            elif parsed.path == "/api/source-text":
                query = parse_qs(parsed.query)
                relative_path = query.get("relative_path", [""])[0]
                self.send_json(read_source_text(relative_path))
            else:
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - simple local dashboard error surface
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            payload = self.read_json_body()
            if parsed.path == "/api/decision":
                self.send_json(upsert_decision(payload))
            elif parsed.path == "/api/holder":
                self.send_json(add_holder(payload), HTTPStatus.CREATED)
            else:
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001 - simple local dashboard error surface
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> int:
    ensure_storage_files()
    parser = argparse.ArgumentParser(description="Run the data source standardization dashboard.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the system browser automatically.")
    args = parser.parse_args()
    host = args.host
    port = args.port

    address = (host, port)
    server = ThreadingHTTPServer(address, DashboardHandler)
    url = f"http://{host}:{port}"
    print("Data Source Standardization Dashboard")
    print(f"Serving: {url}")
    print(f"Input rows: {INPUT_ROWS_PATH}")
    print(f"Holder dictionary: {DICTIONARY_PATH}")
    print(f"Decisions: {DECISIONS_PATH}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
