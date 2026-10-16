#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import os
import re
import time
import pathlib
import threading
from typing import Dict, Iterable, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse, urljoin, unquote


BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def download_pdfs_by_doi(
    doi_to_urls: Dict[str, Iterable[str]],
    output_dir: str,
    max_workers: int = 16,
    timeout: int = 30,
    max_retries_per_url: int = 2,
    fast_pass_timeout: int = 15,
    fast_pass_retries_per_url: int = 1,
    per_host_limit: int = 3,
) -> List[Tuple[str, Optional[str], str, Optional[str], str]]:
    """
    Download PDFs given {doi: [candidate_urls...]}.

    The downloader now uses a two-stage flow:
    1. Fast pass for direct candidate URLs only.
    2. Slower fallback that resolves DOI-like candidates and, if needed, tries Europe PMC.

    Files are saved as <normalized-doi>.pdf (slashes -> '_', etc.).

    Prints one line per DOI when finished:
        [OK]   - DOI
        [EPMC] - DOI
        [FAIL] - DOI

    Returns a list of tuples:
        (doi, saved_filename|None, status, winning_url|None, explanation)

      - doi: original DOI key from doi_to_urls
      - saved_filename: the actual filename used on disk (e.g. "10.1016_j.marpol.2024.105123.pdf"
                        or "10.1016_j.marpol.2024.105123 (1).pdf"), or None if no file was saved
      - status: "OK", "EPMC", or "FAIL"
      - winning_url: URL that succeeded (or None)
      - explanation: human-readable trace of what happened
    """

    outdir = pathlib.Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    thread_local = threading.local()
    host_gate_lock = threading.Lock()
    host_gates: Dict[Tuple[str, str], threading.BoundedSemaphore] = {}

    # -------------------- helpers --------------------

    def _sanitize_filename(name: str) -> str:
        name = re.sub(r"[\\/*?\"<>|:#]", "_", name).strip()
        name = re.sub(r"\s+", " ", name)
        return name[:200] if len(name) > 200 else name

    def _ensure_pdf_extension(name: str) -> str:
        return name if name.lower().endswith(".pdf") else f"{name}.pdf"

    def _unique_path(base_dir: pathlib.Path, filename: str) -> pathlib.Path:
        candidate = base_dir / filename
        if not candidate.exists():
            return candidate
        stem, suffix = candidate.stem, candidate.suffix
        i = 1
        while True:
            alt = base_dir / f"{stem} ({i}){suffix}"
            if not alt.exists():
                return alt
            i += 1

    def _norm_doi_for_filename(doi: str) -> str:
        d = (doi or "").strip()
        d = d.replace("https://doi.org/", "").replace("doi:", "")
        d = d.replace("/", "_")
        return _sanitize_filename(d) or "document"

    def _filename_from_doi(doi: str) -> str:
        return _ensure_pdf_extension(_norm_doi_for_filename(doi))

    def _looks_like_pdf(content_type: Optional[str], final_url: str) -> bool:
        if content_type and content_type.lower().startswith("application/pdf"):
            return True
        return final_url.lower().endswith(".pdf")

    def _save_stream(resp: requests.Response, dest: pathlib.Path, chunk_size: int = 32768):
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)

    def _get_worker_session() -> requests.Session:
        session = getattr(thread_local, "session", None)
        if session is None:
            session = requests.Session()
            pool_size = max(1, max_workers)
            adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            session.headers.update({"User-Agent": BROWSER_UA})
            thread_local.session = session
        return session

    def _get_host_gate(url: str) -> threading.BoundedSemaphore:
        parsed = urlparse(url)
        gate_key = (parsed.scheme.lower(), parsed.netloc.lower())
        with host_gate_lock:
            if gate_key not in host_gates:
                host_gates[gate_key] = threading.BoundedSemaphore(max(1, per_host_limit))
            return host_gates[gate_key]

    def _session_get(session: requests.Session, url: str, **kwargs):
        gate = _get_host_gate(url)
        gate.acquire()
        try:
            return session.get(url, **kwargs)
        finally:
            gate.release()

    def _is_doi_like(candidate: str) -> bool:
        c = (candidate or "").strip()
        return (
            c.lower().startswith("doi:")
            or c.startswith("10.")
            or c.lower().startswith("https://doi.org/")
        )

    def _normalize_direct_candidate(candidate: str) -> Tuple[str, Optional[str], bool]:
        u = (candidate or "").strip()
        if not u:
            return u, "Empty URL", False

        if _is_doi_like(u):
            return u, None, True

        if not urlparse(u).scheme:
            u = "https://" + u

        parsed = urlparse(u)
        if not parsed.netloc or "." not in parsed.netloc:
            if "10." in u:
                return u, None, True
            return u, "Invalid URL/hostname", False

        return u, None, False

    # ---------- DOI/URL normalization ----------

    def _resolve_doi_to_pdf(session: requests.Session, doi_like: str, timeout: int) -> Optional[str]:
        doi = (doi_like or "").strip()
        if doi.lower().startswith("doi:"):
            doi = doi[4:]
        doi_url = doi if doi.lower().startswith("http") else f"https://doi.org/{doi}"

        try:
            r = _session_get(
                session,
                doi_url,
                headers={"User-Agent": BROWSER_UA, "Accept": "application/pdf"},
                allow_redirects=True,
                timeout=timeout,
                stream=False,
            )
        except requests.RequestException:
            return None

        if _looks_like_pdf(r.headers.get("Content-Type", ""), r.url):
            return r.url

        land = r.url
        try:
            r = _session_get(session, land, headers={"User-Agent": BROWSER_UA}, timeout=timeout)
        except requests.RequestException:
            return None

        html = r.text or ""
        m = re.search(r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']', html, re.I)
        if m:
            pdf_url = m.group(1)
            if not urlparse(pdf_url).netloc:
                pdf_url = urljoin(land, pdf_url)
            return pdf_url
        return None

    def _normalize_to_pdfish(session: requests.Session, candidate: str, timeout: int) -> Tuple[str, Optional[str]]:
        u = (candidate or "").strip()
        if not u:
            return u, "Empty URL"
        if u.lower().startswith("doi:") or u.startswith("10.") or u.lower().startswith("https://doi.org/"):
            pdf = _resolve_doi_to_pdf(session, u, timeout)
            return (pdf or u, None if pdf else "Could not resolve DOI to a PDF")
        if not urlparse(u).scheme:
            u = "https://" + u
        parsed = urlparse(u)
        if not parsed.netloc or "." not in parsed.netloc:
            if "10." in u:
                pdf = _resolve_doi_to_pdf(session, u, timeout)
                if pdf:
                    return pdf, None
            return u, "Invalid URL/hostname"
        return u, None

    # ---------- Europe PMC OA fallback ----------

    def _europe_pmc_pdf_candidates(session: requests.Session, doi: str, timeout: int) -> List[str]:
        epmc = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {"query": f"DOI:{doi}", "format": "json"}
        try:
            r = _session_get(session, epmc, params=params, headers={"User-Agent": BROWSER_UA}, timeout=timeout)
            r.raise_for_status()
            js = r.json()
        except requests.RequestException:
            return []

        hits = js.get("resultList", {}).get("result", []) or []
        if not hits:
            return []

        candidates: List[str] = []
        for hit in hits:
            ftl = hit.get("fullTextUrlList", {}).get("fullTextUrl", []) or []
            for ft in ftl:
                if ft.get("documentStyle") == "pdf" and ft.get("url"):
                    candidates.append(ft["url"])

            pmcid = hit.get("pmcid")
            if pmcid:
                if not pmcid.upper().startswith("PMC"):
                    pmcid = "PMC" + str(pmcid)
                candidates += [
                    f"https://europepmc.org/articles/{pmcid}?pdf=render",
                    f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf",
                    f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf",
                ]

        # de-dup preserve order
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    # ---------- network try (returns Response so we can save) ----------

    def _try_single_url(
        session: requests.Session,
        url: str,
        request_timeout: int,
    ) -> Tuple[Optional[requests.Response], Optional[str]]:
        """
        Try to fetch a URL; return (Response|None, error|None).
        Caller is responsible for closing/saving the response on success.
        """
        headers_base = {
            "User-Agent": BROWSER_UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

        last_err = None
        for accept in ("*/*", "application/pdf"):
            headers = dict(headers_base)
            headers["Accept"] = accept
            headers["Referer"] = f"{urlparse(url).scheme}://{urlparse(url).netloc}" if "://" in url else "https://google.com"

            try:
                r = _session_get(
                    session,
                    url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=request_timeout,
                )
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                continue

            if r.status_code == 403:
                last_err = "403 Forbidden"
                r.close()
                continue

            try:
                r.raise_for_status()
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                r.close()
                continue

            ctype = r.headers.get("Content-Type", "")
            if not _looks_like_pdf(ctype, r.url):
                # Try to discover a declared PDF on the page
                html = r.text if not r.raw.closed else ""
                m = re.search(r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']', html, re.I)
                if m:
                    pdf2 = m.group(1)
                    if not urlparse(pdf2).netloc:
                        pdf2 = urljoin(r.url, pdf2)
                    r.close()
                    try:
                        r = _session_get(
                            session,
                            pdf2,
                            headers=headers,
                            stream=True,
                            allow_redirects=True,
                            timeout=request_timeout,
                        )
                        r.raise_for_status()
                    except requests.RequestException as e:
                        last_err = f"{type(e).__name__}: {e}"
                        continue
                    ctype = r.headers.get("Content-Type", "")

            if not _looks_like_pdf(ctype, r.url):
                last_err = f"Unexpected Content-Type: {ctype} at {r.url}"
                r.close()
                continue

            # Success: return open Response for the caller to save/close
            return r, None

        return None, last_err or "Unknown error"

    # ---------- per-DOI worker ----------

    def _process_one_doi(doi: str, candidates: Iterable[str]) -> Tuple[str, Optional[str], str, Optional[str], str]:
        """
        Return (doi, saved_filename|None, status, winning_url|None, explanation). Also saves the PDF.
        """
        fast_pass_failures: List[str] = []
        slow_pass_failures: List[str] = []
        epmc_failures: List[str] = []

        if isinstance(candidates, str):
            candidate_iterable = [candidates]
        else:
            candidate_iterable = candidates

        cand_list = []
        for candidate in candidate_iterable:
            if candidate is None:
                continue
            candidate_text = str(candidate).strip()
            if candidate_text:
                cand_list.append(candidate_text)

        if not cand_list:
            return doi, None, "FAIL", None, "No provided URLs. Europe PMC not attempted."

        session = _get_worker_session()

        # 1) Fast pass for direct URLs only
        fast_candidates = []
        for cand in cand_list:
            url_fast, fast_err, is_doi_like = _normalize_direct_candidate(cand)
            if fast_err:
                fast_pass_failures.append(f"{cand} -> {fast_err}")
                continue
            fast_candidates.append((cand, url_fast, is_doi_like))

        for _original_candidate, fast_url, is_doi_like in fast_candidates:
            if is_doi_like:
                continue

            backoff = 1.0
            last_err = None
            for attempt in range(1, fast_pass_retries_per_url + 1):
                resp, err = _try_single_url(session, fast_url, fast_pass_timeout)
                if resp is not None:
                    dest = _unique_path(outdir, _filename_from_doi(doi))
                    _save_stream(resp, dest)
                    resp.close()
                    saved_filename = dest.name

                    explanation = "Fast pass direct URLs: "
                    if fast_pass_failures:
                        explanation += "; ".join(fast_pass_failures) + "; "
                    explanation += f"SUCCEEDED with {fast_url}"
                    return doi, saved_filename, "OK", fast_url, explanation

                last_err = err or "failed"
                if attempt < fast_pass_retries_per_url:
                    time.sleep(backoff)
                    backoff *= 2
            fast_pass_failures.append(f"{fast_url} -> {last_err or 'failed'}")

        # 2) Slower provided-URL pass, including DOI-like candidates
        for cand in cand_list:
            url_norm, norm_err = _normalize_to_pdfish(session, cand, timeout)
            if norm_err:
                slow_pass_failures.append(f"{cand} -> {norm_err}")
                continue

            backoff = 1.0
            last_err = None
            for attempt in range(1, max_retries_per_url + 1):
                resp, err = _try_single_url(session, url_norm, timeout)
                if resp is not None:
                    dest = _unique_path(outdir, _filename_from_doi(doi))
                    _save_stream(resp, dest)
                    resp.close()
                    saved_filename = dest.name

                    explanation_parts = []
                    if fast_pass_failures:
                        explanation_parts.append("Fast pass failed: " + "; ".join(fast_pass_failures))
                    slow_text = "Slow pass provided URLs: "
                    if slow_pass_failures:
                        slow_text += "; ".join(slow_pass_failures) + "; "
                    slow_text += f"SUCCEEDED with {url_norm}"
                    explanation_parts.append(slow_text)
                    return doi, saved_filename, "OK", url_norm, ". ".join(explanation_parts)

                last_err = err or "failed"
                if attempt < max_retries_per_url:
                    time.sleep(backoff)
                    backoff *= 2
            slow_pass_failures.append(f"{url_norm} -> {last_err or 'failed'}")

        # 3) Europe PMC fallback remains automatic
        epmc_candidates = _europe_pmc_pdf_candidates(session, doi, timeout)
        if not epmc_candidates:
            epmc_failures.append("Europe PMC: no candidates returned")
        else:
            for ec in epmc_candidates:
                resp, err = _try_single_url(session, ec, timeout)
                if resp is not None:
                    dest = _unique_path(outdir, _filename_from_doi(doi))
                    _save_stream(resp, dest)
                    resp.close()
                    saved_filename = dest.name

                    explanation_parts = []
                    if fast_pass_failures:
                        explanation_parts.append("Fast pass failed: " + "; ".join(fast_pass_failures))
                    explanation_parts.append(
                        "Slow pass provided URLs failed: "
                        + ("; ".join(slow_pass_failures) if slow_pass_failures else "none")
                    )
                    explanation_parts.append(f"Europe PMC SUCCEEDED with {ec}")
                    return doi, saved_filename, "EPMC", ec, ". ".join(explanation_parts)
                epmc_failures.append(f"{ec} -> {err or 'failed'}")

        explanation_parts = []
        if fast_pass_failures:
            explanation_parts.append("Fast pass failed: " + "; ".join(fast_pass_failures))
        explanation_parts.append(
            "Slow pass provided URLs failed: "
            + ("; ".join(slow_pass_failures) if slow_pass_failures else "none")
        )
        explanation_parts.append(
            "Europe PMC failed: " + ("; ".join(epmc_failures) if epmc_failures else "not attempted")
        )
        return doi, None, "FAIL", None, ". ".join(explanation_parts)

    # ---------- parallel run ----------

    results: List[Tuple[str, Optional[str], str, Optional[str], str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_one_doi, doi, urls): doi for doi, urls in doi_to_urls.items()}
        for fut in as_completed(futures):
            doi = futures[fut]
            try:
                doi_ret, saved_filename, status, winning_url, explanation = fut.result()
                print(f"[{status}] - {doi_ret}")
                results.append((doi_ret, saved_filename, status, winning_url, explanation))
            except Exception as e:
                print(f"[FAIL] - {doi}")
                results.append((doi, None, "FAIL", None, f"Unhandled exception: {e}"))

    return results


# In[ ]:





# In[ ]:





# In[1]:


import os
import re
import time
import pathlib
from typing import Iterable, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from urllib.parse import urlparse, urljoin, unquote


BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def download_pdfs_for_publications(
    url_groups: Iterable[Iterable[str]],     # <- list of lists
    output_dir: str,
    max_workers: int = 2,
    timeout: int = 30,
    max_retries_per_url: int = 2,
) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
    """
    Download PDFs for a list of publications, where each publication has
    multiple candidate URLs (DOIs or direct links). For each publication,
    try candidates in order until one works.

    Parameters
    ----------
    url_groups : Iterable[Iterable[str]]
        Outer iterable = publications; inner iterable = candidate URLs for that pub.
    output_dir : str
        Directory to save PDFs.
    max_workers : int
        Publications processed in parallel.
    timeout : int
        Per-request timeout.
    max_retries_per_url : int
        Retries for a *single* candidate URL before moving on to the next candidate.

    Returns
    -------
    List[Tuple[Optional[str], Optional[str], Optional[str]]]
        One tuple per publication:
        (winning_url_or_none, saved_path_or_none, error_message_if_all_failed_or_none).

        If a candidate succeeded, error is None.
        If none succeeded, winning_url and saved_path are None and error summarizes per-candidate failures.
    """

    outdir = pathlib.Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------- helpers --------------------

    def _sanitize_filename(name: str) -> str:
        name = re.sub(r"[\\/*?\"<>|:#]", "_", name).strip()
        name = re.sub(r"\s+", " ", name)
        return name[:200]

    def _ensure_pdf_extension(name: str) -> str:
        return name if name.lower().endswith(".pdf") else f"{name}.pdf"

    def _unique_path(base_dir: pathlib.Path, filename: str) -> pathlib.Path:
        candidate = base_dir / filename
        if not candidate.exists():
            return candidate
        stem, suffix = candidate.stem, candidate.suffix
        i = 1
        while True:
            alt = base_dir / f"{stem} ({i}){suffix}"
            if not alt.exists():
                return alt
            i += 1

    def _filename_from_headers(content_disp: Optional[str]) -> Optional[str]:
        if not content_disp:
            return None
        m = re.search(r"filename\*\s*=\s*(?:UTF-8''|utf-8'')?([^;]+)", content_disp)
        if m:
            return _sanitize_filename(unquote(m.group(1)))
        m = re.search(r'filename\s*=\s*"([^"]+)"', content_disp)
        if m:
            return _sanitize_filename(m.group(1))
        m = re.search(r"filename\s*=\s*([^;]+)", content_disp)
        if m:
            return _sanitize_filename(m.group(1).strip())
        return None

    def _filename_from_url(url: str) -> str:
        path = urlparse(url).path
        last = _sanitize_filename(os.path.basename(unquote(path))) or "download"
        return last

    def _looks_like_pdf(content_type: Optional[str], final_url: str) -> bool:
        if content_type and content_type.lower().startswith("application/pdf"):
            return True
        return final_url.lower().endswith(".pdf")

    def _save_stream(resp: requests.Response, dest: pathlib.Path, chunk_size: int = 32768):
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)

    def _resolve_doi_to_pdf(session: requests.Session, doi_like: str, timeout: int) -> Optional[str]:
        """
        Try content negotiation at doi.org; if HTML, parse landing page for <meta name="citation_pdf_url">.
        """
        doi = doi_like.strip()
        if doi.lower().startswith("doi:"):
            doi = doi[4:]
        doi_url = doi if doi.lower().startswith("http") else f"https://doi.org/{doi}"

        # Ask for PDF first
        r = session.get(
            doi_url,
            headers={"User-Agent": BROWSER_UA, "Accept": "application/pdf"},
            allow_redirects=True,
            timeout=timeout,
            stream=False,
        )
        if _looks_like_pdf(r.headers.get("Content-Type", ""), r.url):
            return r.url

        # Not a PDF; fetch landing page and look for citation_pdf_url
        land = r.url
        r = session.get(land, headers={"User-Agent": BROWSER_UA}, timeout=timeout)
        html = r.text or ""
        m = re.search(r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']', html, re.I)
        if m:
            pdf_url = m.group(1)
            if not urlparse(pdf_url).netloc:
                pdf_url = urljoin(land, pdf_url)
            return pdf_url
        return None

    def _normalize_to_pdfish(session: requests.Session, candidate: str, timeout: int) -> Tuple[str, Optional[str]]:
        """
        Convert a candidate (URL or DOI) to a 'best guess' PDF URL.
        Returns (url, None) on success, or (original, reason) if we could not normalize.
        """
        u = (candidate or "").strip()
        if not u:
            return u, "Empty URL"

        # DOI?
        if u.lower().startswith("doi:") or u.startswith("10.") or u.lower().startswith("https://doi.org/"):
            pdf = _resolve_doi_to_pdf(session, u, timeout)
            return (pdf or u, None if pdf else "Could not resolve DOI to a PDF")

        # Add scheme if missing
        if not urlparse(u).scheme:
            u = "https://" + u

        # Hostname sanity
        parsed = urlparse(u)
        if not parsed.netloc or "." not in parsed.netloc:
            if "10." in u:  # mis-pasted DOI-like
                pdf = _resolve_doi_to_pdf(session, u, timeout)
                if pdf:
                    return pdf, None
            return u, "Invalid URL/hostname"

        return u, None

    # -------------------- workers --------------------

    def _try_single_url(session: requests.Session, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Attempt to download a single (normalized) URL with a couple of header tweaks.
        Returns (saved_path|None, error|None).
        """
        headers = {
            "User-Agent": BROWSER_UA,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Referer": f"{urlparse(url).scheme}://{urlparse(url).netloc}" if "://" in url else "https://google.com",
        }

        # Try with Accept */*, then with application/pdf
        for accept in ("*/*", "application/pdf"):
            headers["Accept"] = accept
            r = session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=timeout)
            if r.status_code == 403:
                r.close()
                continue  # switch accept or fail

            try:
                r.raise_for_status()
            except Exception as e:
                r.close()
                return None, f"{type(e).__name__}: {e}"

            ctype = r.headers.get("Content-Type", "")
            if not _looks_like_pdf(ctype, r.url):
                # Maybe we got an HTML page; see if it declares a citation_pdf_url
                html = r.text if not r.raw.closed else ""
                m = re.search(r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']', html, re.I)
                if m:
                    pdf2 = m.group(1)
                    if not urlparse(pdf2).netloc:
                        pdf2 = urljoin(r.url, pdf2)
                    r.close()
                    # second try directly on discovered link
                    r = session.get(pdf2, headers=headers, stream=True, allow_redirects=True, timeout=timeout)
                    try:
                        r.raise_for_status()
                    except Exception as e:
                        r.close()
                        return None, f"{type(e).__name__}: {e}"
                    ctype = r.headers.get("Content-Type", "")

            if not _looks_like_pdf(ctype, r.url):
                # Still not a PDF
                err = f"Unexpected Content-Type: {ctype} at {r.url}"
                r.close()
                return None, err

            # Save
            fname = (_filename_from_headers(r.headers.get("Content-Disposition"))
                     or _filename_from_url(r.url))
            fname = _ensure_pdf_extension(fname)
            dest = _unique_path(outdir, fname)
            _save_stream(r, dest)
            r.close()
            return str(dest), None

        return None, "403 or non-PDF response"

    def _process_one_publication(candidates: Iterable[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Iterate over candidate URLs for a single publication; retry each URL a few times before moving on.
        Returns (winning_url, saved_path, error_or_None).
        """
        errors = []
        cand_list = [c for c in candidates if (c or "").strip()]
        if not cand_list:
            return None, None, "No candidates provided"

        with requests.Session() as s:
            s.headers.update({"User-Agent": BROWSER_UA})

            for cand in cand_list:
                # Normalize (DOI -> PDF URL etc.)
                url_norm, norm_err = _normalize_to_pdfish(s, cand, timeout)
                if norm_err:
                    errors.append(f"{cand} -> {norm_err}")
                    continue

                # Try this candidate with limited retries
                backoff = 1.0
                for attempt in range(1, max_retries_per_url + 1):
                    saved, err = _try_single_url(s, url_norm)
                    if saved:
                        return url_norm, saved, None
                    # failed: record and maybe retry
                    last = f"{cand} -> {err}"
                    if attempt < max_retries_per_url:
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        errors.append(last)

        return None, None, "; ".join(errors) if errors else "All candidates failed"

    # -------------------- pool --------------------

    results: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_one_publication, grp): i for i, grp in enumerate(url_groups, start=1)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                winning_url, saved_path, err = fut.result()
                if err:
                    print(f"[FAIL] publication #{i}\n       -> {err}")
                else:
                    print(f"[OK] publication #{i}\n       -> {winning_url}\n       -> {saved_path}")
                results.append((winning_url, saved_path, err))
            except Exception as e:
                print(f"[FAIL] publication #{i}\n       -> Exception: {e}")
                results.append((None, None, str(e)))
    
    return results


# In[ ]:

