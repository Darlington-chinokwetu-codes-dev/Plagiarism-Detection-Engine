"""
Plagiarism Detection Engine
===========================
EdTech / content-verification tool that compares a *suspicious* document against a
*source* document and flags:

  * verbatim / near-verbatim copying   -> lexical signals (TF-IDF cosine + word-trigram containment)
  * paraphrased copying                -> semantic signal (Sentence-BERT dense embeddings)
  * properly cited quotations          -> detected and excluded from the score

Run locally:   streamlit run app.py
"""
from __future__ import annotations

from contextlib import contextmanager
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import html
import hashlib
import io
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, TypedDict
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
MODEL_OPTIONS = {
    "all-MiniLM-L6-v2 (fast, ~90 MB)": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2 (most accurate, ~420 MB)": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2 (50+ languages)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
MODE_HYBRID = "Hybrid (recommended)"
MODE_SEMANTIC = "Semantic only (SBERT)"
MODE_LEXICAL = "Lexical only (TF-IDF)"
MODES = [MODE_HYBRID, MODE_SEMANTIC, MODE_LEXICAL]

FLAGGED = ("exact", "paraphrase", "similar")
PRIORITY = {"similar": 1, "paraphrase": 2, "exact": 3}
SCORE_WEIGHT = {"exact": 1.0, "paraphrase": 1.0, "similar": 0.5}
STATUS_META = {
    "exact": ("hl-exact", "Verbatim / near-verbatim", "#ef4444"),
    "paraphrase": ("hl-para", "Semantic paraphrase", "#f97316"),
    "similar": ("hl-sim", "Partial similarity", "#eab308"),
    "cited": ("hl-cite", "Cited quotation (excluded)", "#3b82f6"),
}
VERDICT_COLORS = {"Safe": "#22c55e", "Suspicious": "#f59e0b", "Plagiarized": "#ef4444"}
TRANSPARENT = "rgba(0,0,0,0)"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_SEARCH_EXCERPTS = 10
MAX_WORDS_PER_EXCERPT = 16
TAVILY_RESULTS_PER_QUERY = 5
SEARCH_TIME_BUDGET_SECONDS = 300
TAVILY_READ_TIMEOUT_SECONDS = 90
TAVILY_FALLBACK_TIMEOUT_SECONDS = 45
MIN_MATCH_WORDS = 5

# --------------------------------------------------------------------------------------
# Built-in samples (original paragraph + cleverly paraphrased copy)
# --------------------------------------------------------------------------------------
SAMPLE_SOURCE = (
    "The Industrial Revolution, which began in Britain in the late eighteenth century, "
    "transformed the way goods were produced. Handmade products gave way to machine "
    "manufacturing, and factories sprang up in rapidly growing cities. Workers left farms "
    "in search of wages, and urban populations swelled almost overnight. Although "
    "productivity soared, the new factory system also brought dangerous working conditions "
    "and long hours. Over time, public pressure led governments to introduce labor laws "
    "that protected workers and children.\n\n"
    'As one historian put it, "Industrialization rewrote the rules of daily life within a '
    'single generation" (Clarke, 2009). Today, scholars still debate whether the benefits '
    "of industrialization outweighed its social costs."
)

SAMPLE_SUSPICIOUS = (
    "Beginning in Great Britain during the late 1700s, the Industrial Revolution changed "
    "how products were made. Items that had once been crafted by hand were replaced by "
    "machine production, and factories appeared across fast-expanding towns. Many people "
    "abandoned agricultural work to chase wages, and city populations ballooned almost "
    "overnight. Even though output increased dramatically, the factory model introduced "
    "hazardous conditions and exhausting shifts. Over time, public pressure led governments "
    "to introduce labor laws that protected workers and children.\n\n"
    'As one historian put it, "Industrialization rewrote the rules of daily life within a '
    'single generation" (Clarke, 2009). My own view is that the cheap smartphones in modern '
    "pockets owe more to later electronics breakthroughs than to steam engines."
)

# --------------------------------------------------------------------------------------
# Text extraction and sentence segmentation
# --------------------------------------------------------------------------------------
def extract_text(uploaded) -> str:
    """Read text from an uploaded .txt / .md / .pdf / .docx file."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        import docx  # python-docx

        document = docx.Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
    return data.decode("utf-8", errors="ignore")


_ABBREVIATIONS = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "e.g.", "i.e.", "et al.", "vs.", "etc.", "Fig.", "No.", "Inc.", "St."]
_SENT_SPLIT = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["”’\']))\s+(?=[A-Z0-9"“‘\'])')
_QUOTE_RE = re.compile(r'"[^"]{8,}"|“[^”]{8,}”')
_CITE_RE = re.compile(
    r"\([A-Z][^()]{0,60}?\d{4}[a-z]?(?:[,;:][^()]{0,30})?\)"  # (Author, 2020) / (Author et al., 2020, p. 4)
    r"|\[\d+(?:[,\u2013\-]\s*\d+)*\]"  # [1], [1, 2], [3-5]
)


@dataclass
class Sentence:
    para: int
    text: str
    n_words: int
    cited: bool


class SearchExcerpt(TypedDict):
    sentence_number: int
    phrase: str


class SearchSource(TypedDict):
    title: str
    url: str
    description: str
    content: str
    content_kind: str
    matched_excerpts: list[SearchExcerpt]


class SearchResults(TypedDict):
    excerpts: list[SearchExcerpt]
    sources: list[SearchSource]
    requested_count: int
    incomplete: bool
    warnings: list[str]
    snippet_queries: int


def _quoted_fraction(text: str) -> float:
    return sum(len(m.group(0)) for m in _QUOTE_RE.finditer(text)) / max(len(text), 1)


def parse_document(text: str) -> list[Sentence]:
    """Split a document into sentences, remembering paragraphs and cited-quote status."""
    sentences: list[Sentence] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    for p_idx, para in enumerate(paragraphs):
        para = " ".join(para.split())
        for abbr in _ABBREVIATIONS:  # protect abbreviation periods from the splitter
            para = para.replace(abbr, abbr.replace(".", "\u222f"))
        for raw in _SENT_SPLIT.split(para):
            sent = raw.replace("\u222f", ".").strip()
            if not sent:
                continue
            cited = _quoted_fraction(sent) >= 0.5 and bool(_CITE_RE.search(sent))
            sentences.append(Sentence(p_idx, sent, len(re.findall(r"\w+", sent)), cited))
    return sentences


class TavilySearchError(RuntimeError):
    """A provider or network failure that should be shown to the user."""


class TavilyTransientError(TavilySearchError):
    """A temporary failure eligible for one bounded retry."""


def get_tavily_api_key() -> str:
    """Read the Tavily key from the current session or local configuration."""
    api_key = st.session_state.get("tavily_key", "").strip() or os.getenv("TAVILY_API_KEY", "").strip()
    if api_key:
        return api_key
    try:
        return str(st.secrets.get("TAVILY_API_KEY", "")).strip()
    except (KeyError, StreamlitSecretNotFoundError):
        return ""


def build_search_excerpts(text: str, limit: int = MAX_SEARCH_EXCERPTS) -> list[SearchExcerpt]:
    """Select evenly spaced, distinctive sentence excerpts for web search."""
    eligible = [
        (index, sentence)
        for index, sentence in enumerate(parse_document(text))
        if sentence.n_words >= 8
    ]
    if not eligible:
        return []
    if len(eligible) > limit:
        positions = [
            round(index * (len(eligible) - 1) / (limit - 1))
            for index in range(limit)
        ] if limit > 1 else [len(eligible) // 2]
        eligible = [eligible[position] for position in positions]

    excerpts = []
    seen_phrases = set()
    for index, sentence in eligible:
        words = re.findall(r"[\w]+(?:['’][\w]+)*", sentence.text, flags=re.UNICODE)
        phrase = " ".join(words[:MAX_WORDS_PER_EXCERPT])
        normalized = phrase.casefold()
        if normalized and normalized not in seen_phrases:
            seen_phrases.add(normalized)
            excerpts.append({"sentence_number": index + 1, "phrase": phrase})
    return excerpts


def _request_tavily_search(phrase: str, api_key: str, *,
                          include_raw_content: bool = True,
                          timeout: tuple[float, float] = (10, TAVILY_READ_TIMEOUT_SECONDS)) -> requests.Response:
    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            json={"query": phrase, "max_results": TAVILY_RESULTS_PER_QUERY,
                  "search_depth": "basic", "auto_parameters": False,
                  "include_answer": False, "include_raw_content": "text" if include_raw_content else False},
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise TavilyTransientError("Tavily did not respond within the request timeout.") from exc
    except requests.ConnectionError as exc:
        raise TavilyTransientError("The connection to Tavily failed. Check your internet connection.") from exc
    except requests.RequestException as exc:
        raise TavilySearchError(f"Could not reach Tavily ({type(exc).__name__}).") from exc

    if response.status_code in (401, 403):
        raise TavilySearchError("Tavily rejected the API key. Check TAVILY_API_KEY.")
    if response.status_code in (429, 432, 433):
        raise TavilySearchError("Tavily rate limit or quota reached. Please wait or check your plan.")
    if response.status_code in (408, 500, 502, 503, 504):
        raise TavilyTransientError(f"Tavily is temporarily unavailable (HTTP {response.status_code}).")
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TavilySearchError(f"Tavily returned HTTP {response.status_code}.") from exc
    return response


def _parse_tavily_results(response: requests.Response) -> list[dict[str, object]]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise TavilySearchError("Tavily returned an unreadable response.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise TavilySearchError("Tavily returned an unexpected results format.")
    results = payload["results"]
    if not isinstance(results, list):
        raise TavilySearchError("Tavily returned an unexpected results format.")
    return [result for result in results if isinstance(result, dict)]


def _is_web_url(url: str) -> bool:
    parsed_url = urlparse(url)
    return parsed_url.scheme in ("http", "https") and bool(parsed_url.netloc)


def search_published_sources(text: str, api_key: str,
                             progress: Optional[Callable[[int, int, str], None]] = None) -> SearchResults:
    """Retain completed searches; retry temporary failures once with lighter content.

    Stop launching requests after a five-minute soft budget or two consecutive
    failed excerpts. Requests timeouts limit socket inactivity, not total runtime.
    """
    excerpts = build_search_excerpts(text)
    sources: dict[str, SearchSource] = {}
    completed: list[SearchExcerpt] = []
    warnings: list[str] = []
    deadline = time.monotonic() + SEARCH_TIME_BUDGET_SECONDS
    consecutive_failures = 0
    lighter_content = False
    snippet_queries = 0
    stop_scan = False
    for index, excerpt in enumerate(excerpts):
        results = None
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining < 1:
                warnings.append("The scan time budget was reached; remaining searches were skipped.")
                stop_scan = True
                break
            if progress:
                progress(index, len(excerpts),
                         f"Searching phrase {index + 1} of {len(excerpts)}"
                         + (" - retrying with search excerpts only" if attempt else ""))
            try:
                response = _request_tavily_search(
                    excerpt["phrase"], api_key, include_raw_content=not lighter_content,
                    timeout=(min(10, remaining / 3), min(TAVILY_FALLBACK_TIMEOUT_SECONDS if lighter_content else TAVILY_READ_TIMEOUT_SECONDS, remaining * 2 / 3)),
                )
                results = _parse_tavily_results(response)
            except TavilyTransientError as exc:
                if attempt == 0:
                    lighter_content = True
                    time.sleep(min(1, max(0, deadline - time.monotonic())))
                    continue
                warnings.append(f"Phrase {index + 1} could not be searched: {exc}")
            except TavilySearchError as exc:
                warnings.append(str(exc))
                stop_scan = True  # Do not spend more credits on auth/quota/invalid responses.
            break
        if results is None:
            consecutive_failures += 1
            if stop_scan or consecutive_failures >= 2:
                if consecutive_failures >= 2:
                    warnings.append("Stopped after two consecutive failed phrases. Try again when the connection improves.")
                break
            continue
        consecutive_failures = 0
        completed.append(excerpt)
        snippet_queries += int(lighter_content)
        for result in results:
            url = str(result.get("url", "")).strip()
            if not _is_web_url(url):
                continue
            source = sources.setdefault(url, {
                "title": str(result.get("title", "")).strip() or url,
                "url": url,
                "description": str(result.get("content") or "").strip(),
                "content": str(result.get("raw_content") or result.get("content") or "")[:60000],
                "content_kind": "Page text" if result.get("raw_content") else "Search excerpt only",
                "matched_excerpts": [],
            })
            if result.get("raw_content") and source["content_kind"] != "Page text":
                source["content"] = str(result["raw_content"])[:60000]
                source["content_kind"] = "Page text"
            source["matched_excerpts"].append(excerpt)
        if progress:
            progress(index + 1, len(excerpts), f"{len(completed)} of {len(excerpts)} phrases searched successfully")

    if excerpts and not completed:
        detail = warnings[0] if warnings else "Tavily did not return a usable response."
        raise TavilySearchError("No searches completed, so no similarity assessment was made. " + detail)
    if snippet_queries:
        warnings.append("Switched to search excerpts after a slow or failed request. "
                        "Full page text may be unavailable, so matches can be missed. "
                        "Retries may consume additional Tavily credits.")
    ordered_sources = sorted(sources.values(), key=lambda source: len(source["matched_excerpts"]), reverse=True)
    return {"excerpts": completed, "sources": ordered_sources, "requested_count": len(excerpts),
            "incomplete": len(completed) != len(excerpts), "warnings": warnings,
            "snippet_queries": snippet_queries}


# --------------------------------------------------------------------------------------
# Similarity engines
# --------------------------------------------------------------------------------------
def lexical_similarity(a: list[str], b: list[str]) -> np.ndarray:
    """
    Lexical similarity matrix (len(a) x len(b)) = max of
      * TF-IDF (1-2 grams) cosine similarity, and
      * word-trigram containment (share of a's trigrams that also occur in b).
    Containment catches verbatim copying even when the source sentence is much longer.
    """
    if not a or not b:
        return np.zeros((len(a), len(b)))
    corpus = a + b
    try:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit(corpus)
    except ValueError:  # every token was a stop word
        tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(corpus)
    cosine = (tfidf.transform(a) @ tfidf.transform(b).T).toarray()  # rows are L2-normalised

    try:
        counter = CountVectorizer(ngram_range=(3, 3), binary=True, token_pattern=r"(?u)\b\w+\b").fit(corpus)
        ca, cb = counter.transform(a), counter.transform(b)
        overlap = (ca @ cb.T).toarray()
        denom = np.maximum(np.asarray(ca.sum(axis=1)).ravel(), 1)[:, None]
        containment = overlap / denom
    except ValueError:  # no trigrams available
        containment = np.zeros_like(cosine)
    return np.clip(np.maximum(cosine, containment), 0.0, 1.0)


@st.cache_resource(show_spinner="Loading transformer model (first run downloads weights)…")
def load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def semantic_similarity(a: list[str], b: list[str], model) -> np.ndarray:
    """Cosine similarity of Sentence-BERT embeddings (len(a) x len(b))."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    ea = model.encode(a, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    eb = model.encode(b, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.clip(np.asarray(ea) @ np.asarray(eb).T, 0.0, 1.0)


def run_analysis(susp_text: str, src_text: str, mode: str, model_label: str) -> dict:
    """Heavy step: parse, embed and build similarity matrices. Thresholds are applied later."""
    start = time.perf_counter()
    susp, src = parse_document(susp_text), parse_document(src_text)
    a, b = [s.text for s in susp], [s.text for s in src]
    lex = lexical_similarity(a, b)
    sem, note = None, None
    if mode != MODE_LEXICAL and a and b:
        try:
            sem = semantic_similarity(a, b, load_model(MODEL_OPTIONS[model_label]))
        except Exception as exc:  # missing package, offline, out of memory ...
            note = f"Transformer model unavailable ({type(exc).__name__}: {exc}). Fell back to TF-IDF only."
    return {
        "susp": susp, "src": src, "lex": lex, "sem": sem, "note": note,
        "elapsed": time.perf_counter() - start, "model": model_label,
    }


# --------------------------------------------------------------------------------------
# Classification and scoring (cheap; re-run whenever the sidebar thresholds change)
# --------------------------------------------------------------------------------------
@dataclass
class Row:
    idx: int
    para: int
    text: str
    n_words: int
    status: str  # exact | paraphrase | similar | original | cited | short
    score: float
    lex: float
    sem: Optional[float]
    src_idx: Optional[int]


def _classification_scores(lex: np.ndarray, sem: Optional[np.ndarray], mode: str) -> tuple[np.ndarray, str]:
    effective_mode = MODE_LEXICAL if sem is None else mode
    if effective_mode == MODE_SEMANTIC:
        return sem, effective_mode
    if effective_mode == MODE_HYBRID:
        return np.maximum(lex, sem), effective_mode
    return lex, effective_mode


def _sentence_status(lex_score: float, score: float, sem_score: Optional[float],
                     mode: str, cfg: dict) -> str:
    if lex_score >= cfg["exact"]:
        return "exact"
    if mode != MODE_LEXICAL and sem_score is not None and sem_score >= cfg["para"]:
        return "paraphrase"
    if score >= cfg["similar"]:
        return "similar"
    return "original"


def _classify_sentence(index: int, sentence: Sentence, combined: np.ndarray,
                       lex: np.ndarray, sem: Optional[np.ndarray], mode: str, cfg: dict) -> Row:
    if sentence.cited and cfg["ignore_cited"]:
        return Row(index, sentence.para, sentence.text, sentence.n_words, "cited", 0.0, 0.0, None, None)
    if sentence.n_words < cfg["min_words"]:
        return Row(index, sentence.para, sentence.text, sentence.n_words, "short", 0.0, 0.0, None, None)
    if combined.shape[1] == 0:
        return Row(index, sentence.para, sentence.text, sentence.n_words, "original", 0.0, 0.0, None, None)

    source_index = int(np.argmax(combined[index]))
    score = float(combined[index, source_index])
    lex_score = float(lex[index, source_index])
    sem_score = float(sem[index, source_index]) if sem is not None else None
    status = _sentence_status(lex_score, score, sem_score, mode, cfg)
    return Row(index, sentence.para, sentence.text, sentence.n_words, status,
               score, lex_score, sem_score, source_index)


def classify(analysis: dict, mode: str, cfg: dict) -> tuple[list[Row], float]:
    susp, lex, sem = analysis["susp"], analysis["lex"], analysis["sem"]
    combined, effective_mode = _classification_scores(lex, sem, mode)
    rows = [
        _classify_sentence(i, sentence, combined, lex, sem, effective_mode, cfg)
        for i, sentence in enumerate(susp)
    ]
    eligible = [r for r in rows if r.status not in ("cited", "short")]
    total_words = sum(r.n_words for r in eligible)
    weighted = sum(r.n_words * SCORE_WEIGHT[r.status] * r.score for r in eligible if r.status in FLAGGED)
    pct = 100.0 * weighted / total_words if total_words else 0.0
    return rows, min(pct, 100.0)


def verdict_for(pct: float, cfg: dict) -> str:
    if pct < cfg["safe_max"]:
        return "Safe"
    return "Suspicious" if pct < cfg["plag_min"] else "Plagiarized"


# --------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
:root {color-scheme:dark;}
.stApp {background:#10100e;color:#f5f5ed;font-family:'DM Sans',sans-serif;}
/* Hide host controls without removing the sidebar's expand/collapse buttons. */
[data-testid="stAppDeployButton"], [data-testid="stStatusWidget"],
[data-testid="stToolbarActions"] {display:none!important;}
[data-testid="stHeader"] {background:rgba(16,16,14,.95);}
[data-testid="stMainBlockContainer"] {max-width:1240px;padding-top:2rem;padding-bottom:4rem;}
[data-testid="stSidebar"] {background:#151611;border-right:1px solid #303228;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;color:#f5f5ed!important;letter-spacing:-.035em;}
h1 {font-weight:800!important;} h2 {font-size:1.6rem!important;}
.hero {background:radial-gradient(ellipse at 100% 0%,#59311c 0,transparent 55%),linear-gradient(115deg,#1b1c17,#23231b);border:1px solid #45412c;border-radius:24px;padding:38px 40px;margin:10px 0 26px;color:#f5f5ed;position:relative;overflow:hidden;}
.hero::before {content:'';position:absolute;left:0;right:0;top:0;height:4px;background:linear-gradient(90deg,#ff6961,#ffad52 45%,#c5f35b);}
.hero h1 {color:#f5f5ed!important;font-size:2.6rem!important;margin:8px 0;max-width:780px;}
.hero p {color:#c5c6b9;max-width:680px;font-size:1.05rem;line-height:1.7;}
.eyebrow {text-transform:uppercase;letter-spacing:.18em;font-size:.72rem;font-weight:700;color:#c5f35b;}
.hero-tag {display:inline-block;background:#c5f35b12;color:#c5f35b;border:1px solid #c5f35b55;border-radius:30px;padding:6px 12px;font-size:.75rem;margin:10px 6px 0 0;}
.hero-tag:nth-last-child(2) {color:#ffad52;background:#ffad5212;border-color:#ffad5255;}
.hero-tag:last-child {color:#ff938d;background:#ff696112;border-color:#ff696155;}
[data-testid="stMetric"] {background:#1b1c17;border:1px solid #34362b;padding:20px;border-radius:16px;}
[data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;font-weight:800;color:#c5f35b;}
.stButton>button,.stDownloadButton>button {border-radius:10px;min-height:42px;font-weight:600;}
.stButton>button[kind="primary"] {background:#c5f35b!important;border:1px solid #c5f35b!important;color:#10100e!important;}
.stButton>button[kind="primary"]:hover {background:#d9ff88!important;border-color:#d9ff88!important;}
[data-testid="stTextArea"] textarea {background:#1b1c17!important;color:#f5f5ed!important;border-radius:12px;line-height:1.75;}
[data-testid="stFileUploader"] {background:#1b1c17;border-radius:14px;padding:12px;border:1px dashed #6a704d;}
[data-testid="stExpander"] {background:#1b1c17;border-radius:12px;}
[data-baseweb="tab-list"] {gap:25px;border-bottom:1px solid #34362b;margin-bottom:20px;}
[data-baseweb="tab"] {font-weight:600;}
[data-baseweb="tab"][aria-selected="true"] {color:#c5f35b;}
.empty-state {text-align:center;padding:45px 24px;background:#1b1c17;border:1px dashed #555a42;border-radius:18px;margin:20px 0;color:#c5c6b9;}
.empty-state h3 {margin-bottom:10px;}
.workspace-note {background:#202219;border:1px solid #42482d;border-radius:18px;padding:24px;margin-top:12px;}
.workspace-note h3 {font-size:1.15rem!important;}
.workspace-note p {color:#c5c6b9;line-height:1.8;font-size:.92rem;}
.step-line {display:flex;gap:12px;align-items:center;margin:18px 0;color:#e0e1d5;}
.step-number {background:#c5f35b;color:#10100e;border-radius:50%;min-width:30px;height:30px;text-align:center;line-height:30px;font-weight:700;}
.step-line:nth-last-child(2) .step-number {background:#ffad52;}
.step-line:last-child .step-number {background:#ff6961;}
.passage-card {background:#1b1c17;border:1px solid #34362b;border-radius:14px;padding:20px;margin:12px 0;line-height:1.85;}
.passage-label {font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:8px;}
[data-testid="stTextInput"] input {color:#f5f5ed!important;background:#1b1c17!important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.8rem;}
button:focus-visible,a:focus-visible {outline:3px solid #ffad52!important;outline-offset:3px;}
@media(max-width:640px){.hero{padding:24px}.hero h1{font-size:1.8rem!important}}
</style>
"""

VERDICT_TEXT = {
    "Safe": "Likely original — no significant overlap with the source was detected.",
    "Suspicious": "Review recommended — notable overlap with the source. Check the highlighted passages.",
    "Plagiarized": "High overlap — strong evidence of verbatim copying or close paraphrasing.",
}


def dashboard_palette() -> dict[str, str]:
    if st.session_state.get("dark_mode", True):
        return {"background": "#10100e", "surface": "#1b1c17", "text": "#f5f5ed",
                "muted": "#b7b9ab", "border": "#34362b", "lime": "#c5f35b",
                "orange": "#ffad52", "red": "#ff6961", "template": "plotly_dark"}
    return {"background": "#f7f8f2", "surface": "#ffffff", "text": "#202219",
            "muted": "#59604c", "border": "#d8ddcc", "lime": "#4d700c",
            "orange": "#995000", "red": "#bd3029", "template": "plotly_white"}


def dashboard_css() -> str:
    """Per-session styling; never mutate global Streamlit theme configuration."""
    css = CSS
    palette = dashboard_palette()
    if not st.session_state.get("dark_mode", True):
        replacements = {
            "color-scheme:dark": "color-scheme:light",
            "rgba(16,16,14,.95)": "rgba(247,248,242,.95)",
            "#10100e": "#f7f8f2", "#1b1c17": "#ffffff", "#151611": "#eef1e6",
            "#f5f5ed": "#202219", "#303228": "#d8ddcc", "#34362b": "#d8ddcc",
            "#45412c": "#e4d7bf", "#59311c": "#ffdec4", "#23231b": "#fff9ef",
            "#c5c6b9": "#59604c", "#202219": "#f0f4e5", "#42482d": "#d8ddcc",
            "#e0e1d5": "#414833", "#555a42": "#acb79a", "#6a704d": "#acb79a",
        }
        # One pass avoids replacing newly inserted values a second time.
        css = re.sub("|".join(re.escape(value) for value in replacements),
                     lambda match: replacements[match.group()], css)
        css += """<style>
        .eyebrow,[data-testid="stMetricValue"],[data-baseweb="tab"][aria-selected="true"] {color:#4d700c!important;}
        .hero-tag {color:#4d700c;border-color:#9cba64;background:#edf6d8;}
        .hero-tag:nth-last-child(2) {color:#995000;border-color:#e3b57b;background:#fff1df;}
        .hero-tag:last-child {color:#bd3029;border-color:#e2a29c;background:#fff0ee;}
        .step-number,.stButton>button[kind="primary"] {color:#10100e!important;}
        </style>"""
    variables = ";".join(f"--tw-{key}:{value}" for key, value in palette.items() if key != "template")
    return css + "<style>:root{" + variables + "}" + """
    /* Native controls have their own Streamlit theme; match this session's palette. */
    [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"], [data-testid="stExpander"] summary,
    [data-baseweb="tab"], [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
        color:var(--tw-text);
    }
    [data-testid="stCaptionContainer"] {color:var(--tw-muted);}
    [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
    [data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
    [data-baseweb="select"] > div, [data-baseweb="popover"], [role="listbox"],
    [data-testid="stFileUploaderDropzone"], [data-testid="stTooltipContent"] {
        background:var(--tw-surface)!important;color:var(--tw-text)!important;border-color:var(--tw-border)!important;
    }
    [data-baseweb="input"] button,[data-testid="stTextInput"] button,
    [data-baseweb="select"] svg {color:var(--tw-muted)!important;fill:currentColor;}
    input::placeholder, textarea::placeholder {color:var(--tw-muted)!important;opacity:1;}
    input,textarea {caret-color:var(--tw-text);}
    [data-testid="stBaseButton-secondary"], [data-testid="stBaseLinkButton-secondary"],
    [data-testid="stBaseButton-headerNoPadding"] {
        background:var(--tw-surface);color:var(--tw-text);border-color:var(--tw-border);
    }
    [data-testid="stExpandSidebarButton"] span,[data-testid="stCollapseSidebarButton"] span {color:var(--tw-text)!important;}
    [data-testid="stAlert"] {background:var(--tw-surface);color:var(--tw-text);border:1px solid var(--tw-border);border-left:3px solid var(--tw-orange);}
    [data-testid="stMarkdownContainer"] a {color:var(--tw-lime);}
    [data-testid="stMetricLabel"], [data-testid="stSliderTickBarMin"],
    [data-testid="stSliderTickBarMax"] {color:var(--tw-muted);}
    [data-testid="stSliderThumbValue"] {color:var(--tw-lime);}
    [data-testid="stSelectbox"] [role="combobox"] {color:var(--tw-text);background:var(--tw-surface);}
    [data-testid="stTable"] {color:var(--tw-text);background:var(--tw-surface);}
    [data-testid="stTable"] th,[data-testid="stTable"] td {border-color:var(--tw-border);}
    .theme-table {overflow:auto;max-height:480px;border:1px solid var(--tw-border);border-radius:12px;background:var(--tw-surface);}
    .theme-table table {width:100%;border-collapse:collapse;color:var(--tw-text);font-size:.85rem;}
    .theme-table th,.theme-table td {padding:12px 16px;text-align:left;border-bottom:1px solid var(--tw-border);min-width:110px;}
    .theme-table th {background:var(--tw-surface);position:sticky;top:0;white-space:nowrap;}
    </style>"""


def dashboard_table(frame: pd.DataFrame) -> None:
    # An escaped HTML table follows the session theme, unlike a canvas-based grid.
    st.markdown('<div class="theme-table">' + frame.to_html(index=False, escape=True, border=0) + '</div>',
                unsafe_allow_html=True)


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_suspicious(rows: list[Row], src: list[Sentence]) -> str:
    paras: dict[int, list[str]] = {}
    for r in rows:
        if r.status in STATUS_META:
            cls, label, _ = STATUS_META[r.status]
            if r.status == "cited":
                tip = f"{label}: quoted and cited, so it is not counted."
            else:
                tip = f"{label} · score {r.score:.0%} · matches source sentence #{(r.src_idx or 0) + 1}: {src[r.src_idx].text[:160]}"
            piece = f'<span class="hl {cls}" title="{_esc(tip)}">{_esc(r.text)}<sup>{r.idx + 1}</sup></span>'
        else:
            piece = _esc(r.text)
        paras.setdefault(r.para, []).append(piece)
    body = "".join(f"<p>{' '.join(parts)}</p>" for parts in paras.values())
    return f'<div class="docbox">{body or "<p><em>Nothing to display.</em></p>"}</div>'


def render_source(rows: list[Row], src: list[Sentence]) -> str:
    matches: dict[int, list[Row]] = {}
    for r in rows:
        if r.status in FLAGGED and r.src_idx is not None:
            matches.setdefault(r.src_idx, []).append(r)
    paras: dict[int, list[str]] = {}
    for j, s in enumerate(src):
        if j in matches:
            best = max(matches[j], key=lambda r: (PRIORITY[r.status], r.score))
            cls, label, _ = STATUS_META[best.status]
            ids = ", ".join(str(r.idx + 1) for r in matches[j])
            tip = f"{label} · reused by suspicious sentence(s) #{ids} · best score {best.score:.0%}"
            piece = f'<span class="hl {cls}" title="{_esc(tip)}">{_esc(s.text)}<sup>←{ids}</sup></span>'
        else:
            piece = _esc(s.text)
        paras.setdefault(s.para, []).append(piece)
    body = "".join(f"<p>{' '.join(parts)}</p>" for parts in paras.values())
    return f'<div class="docbox">{body or "<p><em>Nothing to display.</em></p>"}</div>'


def gauge_figure(pct: float, verdict: str, cfg: dict) -> go.Figure:
    color = VERDICT_COLORS[verdict]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pct, 1),
            number={"suffix": "%", "font": {"size": 54}},
            title={"text": "Similarity score", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, cfg["safe_max"]], "color": "rgba(34,197,94,.30)"},
                    {"range": [cfg["safe_max"], cfg["plag_min"]], "color": "rgba(245,158,11,.30)"},
                    {"range": [cfg["plag_min"], 100], "color": "rgba(239,68,68,.30)"},
                ],
                "threshold": {"line": {"color": color, "width": 5}, "thickness": 0.85, "value": pct},
            },
        )
    )
    fig.update_layout(height=300, margin={"l": 25, "r": 25, "t": 60, "b": 10}, paper_bgcolor=TRANSPARENT)
    return fig


def composition_figure(counts: Counter) -> go.Figure:
    segments = [
        ("Verbatim", counts["exact"], "#ef4444"),
        ("Paraphrase", counts["paraphrase"], "#f97316"),
        ("Partial", counts["similar"], "#eab308"),
        ("Cited (excluded)", counts["cited"], "#3b82f6"),
        ("Original", counts["original"] + counts["short"], "#22c55e"),
    ]
    fig = go.Figure()
    for label, value, color in segments:
        fig.add_bar(y=[""], x=[value], name=label, orientation="h", marker_color=color, hovertemplate=f"{label}: %{{x}} sentences<extra></extra>")
    fig.update_layout(
        barmode="stack", height=120, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
        legend={"orientation": "h", "y": -0.4}, xaxis={"visible": False}, yaxis={"visible": False},
    )
    return fig


def results_dataframe(rows: list[Row], src: list[Sentence]) -> pd.DataFrame:
    labels = {k: v[1] for k, v in STATUS_META.items()} | {"original": "Original", "short": "Too short (skipped)"}
    return pd.DataFrame(
        {
            "#": [r.idx + 1 for r in rows],
            "Suspicious sentence": [r.text for r in rows],
            "Verdict": [labels[r.status] for r in rows],
            "Score": [round(r.score, 3) for r in rows],
            "Lexical": [round(r.lex, 3) for r in rows],
            "Semantic": [None if r.sem is None else round(r.sem, 3) for r in rows],
            "Best source match": [src[r.src_idx].text if r.src_idx is not None and r.status in FLAGGED else "" for r in rows],
        }
    )


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------
def load_samples() -> None:
    st.session_state["susp_text"] = SAMPLE_SUSPICIOUS
    st.session_state.pop("search_results", None)


def clear_all() -> None:
    st.session_state["susp_text"] = ""
    st.session_state["up_n"] = st.session_state.get("up_n", 0) + 1
    st.session_state.pop("document_sig", None)
    st.session_state.pop("search_results", None)


def sidebar_settings() -> None:
    with st.sidebar:
        st.markdown("## Traceword")
        st.caption("YOUR WRITING. CLEARER EVIDENCE.")
        st.toggle("Dark mode", value=True, key="dark_mode",
                  help="Switch off for light mode. Your choice stays for this session.")
        st.divider()
        st.subheader("Your workspace")
        st.write("Find the sources. Understand the overlap. Make an informed review.")
        with st.expander("Connect Tavily", expanded=not bool(get_tavily_api_key())):
            st.text_input("Tavily API key", type="password", key="tavily_key",
                          help="Kept in this browser session. Paste your key and press Enter.")
            st.link_button("Get a Tavily API key", "https://app.tavily.com")
        st.caption("Key available - ready to scan" if get_tavily_api_key() else "Add your key to start scanning")
        st.caption("Allow about 5 minutes for a scan. Slow requests get one retry; progress appears below your document.")
        st.divider()
        st.slider("TF-IDF review threshold", 0.2, 0.9, 0.5, 0.05, key="review_threshold")
        st.caption("Lower values flag more passages. This is a review setting, not a calibrated plagiarism cutoff.")
        st.divider()
        st.caption("Powered by Tavily + TF-IDF")
        st.caption("Only sampled phrases are sent to Tavily. History stores scan summaries locally, without your document or API key.")


def input_pane() -> None:
    st.subheader("📄 Document to check")
    uploaded = st.file_uploader(
        "Upload a file (.txt, .md, .pdf, .docx)", type=["txt", "md", "pdf", "docx"],
        key=f"up_document_{st.session_state.get('up_n', 0)}",
    )
    if uploaded is not None:
        file_sig = f"{uploaded.name}:{hashlib.sha256(uploaded.getvalue()).hexdigest()}"
        if st.session_state.get("document_sig") != file_sig:
            try:
                st.session_state["susp_text"] = extract_text(uploaded)
                st.session_state["document_sig"] = file_sig
                st.session_state.pop("search_results", None)
            except Exception as exc:
                st.error(f"Could not read {uploaded.name}: {exc}")
    st.text_area(
        "Or paste your text here", key="susp_text", height=260,
        placeholder="Paste the document text you want to check…",
    )


def compare_web_sources(text: str, sources: list[SearchSource]) -> list[dict]:
    """Fit word unigram/bigram TF-IDF jointly; retrieve each sentence's best passage.

    Bound source passages and compare one input sentence at a time to avoid a
    document-by-corpus dense matrix. Scores are cosine similarity, not probability.
    """
    sentences = parse_document(text)
    passages = []
    for source in sources:
        # Split long provider paragraphs into overlapping windows as well.
        for sentence in parse_document(source["content"]):
            words = sentence.text.split()
            for start in range(0, len(words), 60):
                passage = " ".join(words[start:start + 90])
                if len(passage.split()) >= 8:
                    passages.append((passage, source))
                if start + 90 >= len(words):
                    break
            if len(passages) >= 10000:
                break
        if len(passages) >= 10000:
            break
    if not passages:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True,
                                sublinear_tf=True, max_features=100000)
    try:
        matrix = vectorizer.fit_transform([s.text for s in sentences] + [p[0] for p in passages])
    except ValueError:
        return []
    reference = matrix[len(sentences):]
    rows = []
    for index, sentence in enumerate(sentences):
        scores = (matrix[index] @ reference.T).toarray().ravel()
        best = int(scores.argmax())
        passage, source = passages[best]
        score = float(np.clip(scores[best], 0, 1))
        rows.append({"Sentence": index + 1, "Document passage": sentence.text,
                     "TF-IDF cosine": round(score, 4), "Words": sentence.n_words,
                     "Citation detected": sentence.cited,
                     "Source passage": passage if score > 0 else "",
                     "Source URL": source["url"] if score > 0 else "",
                     "Evidence": source["content_kind"] if score > 0 else "No overlap"})
    return rows


def detected_word_overlap(text: str, sources: list[SearchSource]) -> dict:
    """Find normalized five-word runs and count the union of matched input words.

    Only index phrases from this document. Remember one source occurrence per
    phrase (preferring page text), avoiding a quadratic all-pairs comparison.
    Character offsets retain the original document's punctuation and spacing.
    """
    tokens = list(re.finditer(r"\w+", text))
    words = [token.group().casefold() for token in tokens]
    wanted = {tuple(words[i:i + MIN_MATCH_WORDS]) for i in range(len(words) - MIN_MATCH_WORDS + 1)}
    hits = {}
    source_tokens = {}
    usable_sources = 0
    ordered = sorted(sources, key=lambda source: source["content_kind"] != "Page text")
    for source_id, source in enumerate(ordered):
        content = source["content"][:60000]
        parts = list(re.finditer(r"\w+", content))
        if len(parts) < MIN_MATCH_WORDS:
            continue
        usable_sources += 1
        source_tokens[source_id] = (source, content, parts)
        normalized = [part.group().casefold() for part in parts]
        for index in range(len(parts) - MIN_MATCH_WORDS + 1):
            phrase = tuple(normalized[index:index + MIN_MATCH_WORDS])
            if phrase in wanted and phrase not in hits:
                hits[phrase] = (source_id, index)
        if wanted and len(hits) == len(wanted):
            break

    runs = []
    current = None
    for index in range(len(words) - MIN_MATCH_WORDS + 1):
        hit = hits.get(tuple(words[index:index + MIN_MATCH_WORDS]))
        if hit is None:
            current = None
            continue
        source_id, source_index = hit
        if (current is not None and current["source_id"] == source_id
                and current["doc_end"] == index + MIN_MATCH_WORDS - 1
                and current["source_end"] == source_index + MIN_MATCH_WORDS - 1):
            current["doc_end"] += 1
            current["source_end"] += 1
        else:
            current = {"doc_start": index, "doc_end": index + MIN_MATCH_WORDS,
                       "source_id": source_id, "source_start": source_index,
                       "source_end": source_index + MIN_MATCH_WORDS}
            runs.append(current)

    merged = []
    matches = []
    for run in runs:
        start, end = run["doc_start"], run["doc_end"]
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
        source, content, parts = source_tokens[run["source_id"]]
        source_start, source_end = run["source_start"], run["source_end"]
        context_start = parts[max(0, source_start - 8)].start()
        context_end = parts[min(len(parts) - 1, source_end + 7)].end()
        matches.append({"document_start": tokens[start].start(), "document_end": tokens[end - 1].end(),
                        "document_passage": text[tokens[start].start():tokens[end - 1].end()],
                        "source_url": source["url"], "source_title": source["title"],
                        "evidence": source["content_kind"],
                        "source_context": content[context_start:context_end],
                        "source_highlight": [parts[source_start].start() - context_start,
                                             parts[source_end - 1].end() - context_start]})
    matched_words = sum(end - start for start, end in merged)
    return {"percentage": 100 * matched_words / len(tokens) if tokens and usable_sources else None,
            "matched_words": matched_words, "total_words": len(tokens), "minimum_run_words": MIN_MATCH_WORDS,
            "spans": [[tokens[start].start(), tokens[end - 1].end()] for start, end in merged],
            "matches": matches}


def highlight_word_spans(text: str, spans: list[list[int]]) -> str:
    """Escape untrusted document/source text, then mark verified character spans."""
    pieces = []
    position = 0
    for start, end in spans:
        pieces.append(_esc(text[position:start]))
        pieces.append('<mark style="background:#ffc2b8;color:#49100a;border-bottom:2px solid #bd3029;'
                      'border-radius:3px;padding:1px 0" title="Matching wording in a retrieved source">'
                      + _esc(text[start:end]) + '</mark>')
        position = end
    pieces.append(_esc(text[position:]))
    return '<div class="passage-card" style="white-space:pre-wrap">' + ''.join(pieces) + '</div>'


def show_copy_estimate(text: str, overlap: dict, metadata: dict) -> None:
    st.header("Your plagiarism estimate")
    provisional = metadata.get("incomplete", False)
    label = "Estimated plagiarism (partial scan)" if provisional else "Estimated plagiarism"
    score, count, size = st.columns(3)
    score.metric(label, f'{overlap["percentage"]:.1f}%' if overlap["percentage"] is not None else "Unable to estimate")
    count.metric("Matching words", overlap["matched_words"] if overlap["percentage"] is not None else "N/A")
    size.metric("Document words", overlap["total_words"])
    st.caption("This estimate measures matching wording in retrieved sources: matching words / all document words. "
               "Runs of at least five words count, ignoring case and punctuation; overlapping matches count once. "
               "Quotations, citations and common phrases can also match. Review the sources before deciding whether "
               "a passage is plagiarised. A zero score does not prove originality.")
    if metadata.get("snippet_queries"):
        st.caption("Some searches used excerpts without page text. The estimate may miss matches.")
    if overlap["percentage"] is None:
        st.info("No usable source text was retrieved, so a plagiarism percentage cannot be estimated.")
        return
    with st.expander("Your document — matching words highlighted", expanded=True):
        st.caption("Red highlights show the exact word sequences counted in the estimate. Unhighlighted text has no detected match in the retrieved content.")
        st.markdown(highlight_word_spans(text, overlap["spans"]), unsafe_allow_html=True)
    if overlap["matches"]:
        chosen = st.selectbox("Inspect a matching passage", range(len(overlap["matches"])),
                              format_func=lambda i: f'{i + 1}. {overlap["matches"][i]["document_passage"][:85]}')
        match = overlap["matches"][chosen]
        left, right = st.columns(2)
        with left:
            st.markdown("**In your document**")
            st.markdown(highlight_word_spans(match["document_passage"], [[0, len(match["document_passage"])]]), unsafe_allow_html=True)
        with right:
            st.markdown("**In the retrieved source**")
            st.markdown(highlight_word_spans(match["source_context"], [match["source_highlight"]]), unsafe_allow_html=True)
            st.caption(match["source_title"] + " · " + match["evidence"])
            st.link_button("Open matching source", match["source_url"])
    else:
        st.info("No matching runs of five or more words were detected. Similar wording may still appear in the TF-IDF review below.")


def style_dashboard_chart(fig: go.Figure) -> go.Figure:
    """Keep plots readable in the selected theme, including exports."""
    palette = dashboard_palette()
    fig.update_layout(
        template=palette["template"], paper_bgcolor=palette["surface"], plot_bgcolor=palette["surface"],
        font=dict(color=palette["text"], family="Arial, sans-serif"),
        colorway=[palette["lime"], palette["orange"], palette["red"]],
        hoverlabel=dict(bgcolor=palette["background"], font_color=palette["text"]),
    )
    fig.update_xaxes(gridcolor=palette["border"], zerolinecolor=palette["muted"])
    fig.update_yaxes(gridcolor=palette["border"], zerolinecolor=palette["muted"])
    return fig


def show_tfidf_results(text: str, sources: list[SearchSource], search_metadata: Optional[dict] = None) -> None:
    palette = dashboard_palette()
    overlap = detected_word_overlap(text, sources)
    show_copy_estimate(text, overlap, search_metadata or {})
    st.header("Your evidence, at a glance")
    rows = compare_web_sources(text, sources)
    if not rows:
        st.info("No source passages of at least eight words are available for the separate TF-IDF review.")
        st.download_button("Download evidence report (JSON)",
                           json.dumps({"search": search_metadata or {}, "word_overlap": overlap, "sentences": []}, indent=2),
                           file_name="similarity-report.json", mime="application/json")
        return
    threshold = st.session_state.get("review_threshold", 0.5)
    eligible = [r for r in rows if r["Words"] >= 8]
    flagged = [r for r in eligible if r["TF-IDF cosine"] >= threshold]
    total_words = sum(r["Words"] for r in eligible)
    coverage = 100 * sum(r["Words"] for r in flagged) / max(total_words, 1)
    left, right = st.columns(2)
    left.metric("Text above review threshold", f"{coverage:.1f}%" if total_words else "N/A")
    right.metric("Sentences to review", f"{len(flagged)} / {len(eligible)}")
    st.caption("Word-weighted coverage of sentences with at least eight words, against retrieved text only. "
               "This is not a plagiarism percentage. The threshold is an uncalibrated review setting. "
               "TF-IDF measures shared wording and can miss paraphrases. Citation detection is heuristic; "
               "citations remain visible for human review. Source text is capped at 60,000 characters per page "
               "and 10,000 passages overall.")
    with st.expander("Explore sentence similarity", expanded=True):
        fig = go.Figure(go.Bar(x=[r["Sentence"] for r in rows],
                              y=[r["TF-IDF cosine"] for r in rows],
                              marker_color=[palette["red"] if r in flagged else palette["lime"] for r in rows]))
        fig.add_hline(y=threshold, line_dash="dot", line_color=palette["orange"], annotation_text="Review threshold")
        fig.update_layout(height=260, xaxis_title="Document sentence", yaxis=dict(title="TF-IDF cosine", range=[0, 1]),
                          margin=dict(l=20,r=20,t=30,b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(style_dashboard_chart(fig), width="stretch", theme=None)
    view = st.radio("Passage view", ["Needs review", "All passages"], horizontal=True)
    displayed = flagged if view == "Needs review" else rows
    if not displayed:
        st.info("No passages crossed your review threshold. You can still explore all passages and sources below.")
    for row in displayed:
        if row["Words"] < 8:
            color, label = palette["muted"], "Short sentence — not scored in coverage"
        elif row["TF-IDF cosine"] >= threshold:
            color, label = palette["red"], "Review wording overlap"
        else:
            color, label = palette["lime"], "Below review threshold"
        st.markdown(f'<div class="passage-card" style="border-left:4px solid {color}">'
                    f'<div class="passage-label" style="color:{color}">{row["Sentence"]}. {_esc(label)}</div>{_esc(row["Document passage"])}</div>',
                    unsafe_allow_html=True)
        if row in flagged:
            st.caption(f'TF-IDF cosine: {row["TF-IDF cosine"]:.3f} · {row["Evidence"]}'
                       + (" · Possible cited quotation" if row["Citation detected"] else ""))
            st.text(row["Source passage"])
            st.link_button(f'View source for sentence {row["Sentence"]}', row["Source URL"])
    frame = pd.DataFrame(rows)
    with st.expander("All sentence scores and evidence"):
        dashboard_table(frame.drop(columns=["Words"]))
    report = {"search": search_metadata or {}, "word_overlap": overlap, "sentences": rows}
    st.download_button("Download evidence report (JSON)", json.dumps(report, indent=2),
                       file_name="similarity-report.json", mime="application/json")


def show_results(search_results: dict[str, object]) -> None:
    completed = len(search_results["excerpts"])
    requested = search_results.get("requested_count", completed)
    if search_results.get("incomplete"):
        st.warning(f"Partial scan: {completed} of {requested} phrases were searched successfully. "
                   "The evidence below covers only those completed searches.")
    for warning in search_results.get("warnings", []):
        st.warning(warning)
    metadata = {"successful_queries": completed, "requested_queries": requested,
                "incomplete": search_results.get("incomplete", False),
                "snippet_queries": search_results.get("snippet_queries", 0),
                "warnings": search_results.get("warnings", []),
                "review_threshold": st.session_state.get("review_threshold", 0.5)}
    show_tfidf_results(st.session_state["susp_text"], search_results["sources"], metadata)
    excerpts = search_results["excerpts"]
    sources = search_results["sources"]
    st.markdown("---")
    st.header("🌐 Web search results")
    st.warning(
        "These are candidate sources returned for sampled excerpts, not a definitive plagiarism verdict. "
        "Tavily cannot access every publication, paywall, or unindexed page."
    )
    st.metric("Phrases searched successfully", f"{completed} / {requested}")
    if not sources:
        st.info("No candidate pages were returned for these excerpts. This does not prove the document is original.")
        return

    st.metric("Candidate source pages", len(sources))
    for index, source in enumerate(sources, 1):
        with st.expander(f"{index:02d} / {source['title']}"):
            st.caption(urlparse(source["url"]).netloc + " | " + source.get("content_kind", "Search evidence"))
            st.write(source["description"])
            st.link_button("Read original source", source["url"])
            st.caption("Search phrases that discovered this page; these are not verified matches.")
            for excerpt in source["matched_excerpts"]:
                st.text(f"Sentence {excerpt['sentence_number']}: {excerpt['phrase']}")


HISTORY_DB = Path(__file__).parent / ".local" / "scan_history.sqlite3"


@contextmanager
def history_connection():
    HISTORY_DB.parent.mkdir(exist_ok=True)
    connection = sqlite3.connect(HISTORY_DB)
    try:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY, created TEXT, name TEXT, words INTEGER, sources INTEGER, queries INTEGER, coverage REAL, flagged INTEGER, threshold REAL)")
            # Preserve existing history; mark records written before status tracking.
            columns = {row[1] for row in connection.execute("PRAGMA table_info(scans)")}
            for name, declaration in [("status", "TEXT NOT NULL DEFAULT 'legacy'"),
                                      ("requested", "INTEGER"), ("snippet_queries", "INTEGER NOT NULL DEFAULT 0"),
                                      ("overlap_percent", "REAL"), ("overlap_words", "INTEGER"), ("word_total", "INTEGER")]:
                if name not in columns:
                    connection.execute(f"ALTER TABLE scans ADD COLUMN {name} {declaration}")
            yield connection
    finally:
        connection.close()


def save_scan(text, results, name, threshold):
    overlap = detected_word_overlap(text, results["sources"])
    rows = compare_web_sources(text, results["sources"])
    eligible = [r for r in rows if r["Words"] >= 8]
    flagged = [r for r in eligible if r["TF-IDF cosine"] >= threshold]
    denominator = sum(r["Words"] for r in eligible)
    coverage = 100 * sum(r["Words"] for r in flagged) / denominator if denominator else None
    with history_connection() as connection:
        connection.execute("INSERT INTO scans (created,name,words,sources,queries,coverage,flagged,threshold,status,requested,snippet_queries,overlap_percent,overlap_words,word_total) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (datetime.now(timezone.utc).isoformat(), name.strip() or "Untitled scan",
                            len(text.split()), len(results["sources"]), len(results["excerpts"]),
                            coverage, len(flagged), threshold,
                            "partial" if results.get("incomplete") else "complete",
                            results.get("requested_count", len(results["excerpts"])), results.get("snippet_queries", 0),
                            overlap["percentage"], overlap["matched_words"], overlap["total_words"]))


def show_history():
    palette = dashboard_palette()
    st.subheader("Your writing, over time")
    st.caption("Scan summaries saved on this computer. Shared by users of this local app. Documents, source passages and keys are not stored.")
    try:
        with history_connection() as connection:
            frame = pd.read_sql_query("SELECT * FROM scans ORDER BY id DESC", connection)
    except sqlite3.Error:
        st.error("History is unavailable. You can still run a new scan.")
        return
    if frame.empty:
        st.markdown('<div class="empty-state"><h3>Your next scan starts the story.</h3>'
                    'Completed scans will appear here with trends and source counts. No sample data, just your work.</div>', unsafe_allow_html=True)
        return
    period = st.selectbox("Time range", ["All time", "Last 7 days", "Last 30 days"], key="history_period")
    if period != "All time":
        days = 7 if period == "Last 7 days" else 30
        dates = pd.to_datetime(frame.created, utc=True)
        frame = frame[dates >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
    if frame.empty:
        st.info("No scans in this time range. Choose All time to see your earlier work.")
        return
    a, b, c = st.columns(3)
    a.metric("Saved scans", len(frame))
    b.metric("Words reviewed", f"{int(frame.words.sum()):,}")
    c.metric("Source results across scans", int(frame.sources.sum()))
    st.caption("Estimates measure matching word sequences in retrieved sources. Different documents and source availability affect comparisons. Partial scans, missing evidence and older scans without word-level estimates appear as gaps. TF-IDF sentence coverage is retained separately in the table.")
    chronological = frame.sort_values("id")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig = go.Figure(go.Scatter(x=chronological.id, y=chronological.overlap_percent.where(chronological.status != "partial"), mode="lines+markers",
                                  line=dict(color=palette["lime"], width=3), marker=dict(size=9),
                                  text=chronological.name, connectgaps=False,
                                  hovertemplate="%{text}<br>Matching wording: %{y:.1f}%<extra></extra>"))
        fig.update_layout(title="Estimated plagiarism across scans", yaxis=dict(title="Matching words (%)", range=[0,100]), xaxis_title="Scan number", height=300)
        st.plotly_chart(style_dashboard_chart(fig), width="stretch", theme=None)
    with chart_right:
        fig = go.Figure(go.Bar(x=chronological.id, y=chronological.sources, marker_color=palette["orange"],
                              text=chronological.name, hovertemplate="%{text}<br>Sources: %{y}<extra></extra>"))
        fig.update_layout(title="Sources discovered", xaxis_title="Scan number", yaxis_title="Candidate pages", height=300)
        st.plotly_chart(style_dashboard_chart(fig), width="stretch", theme=None)
    query = st.text_input("Find a scan", placeholder="Search by scan name")
    filtered = frame[frame.name.str.contains(query, case=False, regex=False)]
    dashboard_table(filtered.rename(columns={"created":"Time (UTC)", "name":"Scan", "coverage":"TF-IDF sentence coverage (%)", "threshold":"Review threshold", "status":"Status", "queries":"Successful queries", "requested":"Requested queries", "snippet_queries":"Queries without page text", "overlap_percent":"Estimated plagiarism (%)", "overlap_words":"Matching words", "word_total":"Total document words"}).drop(columns=["id"]))
    st.download_button("Export filtered history (JSON)", filtered.to_json(orient="records", indent=2), "scan-history.json", "application/json")
    with st.expander("Manage history"):
        confirmed = st.checkbox("Delete all locally saved scan summaries")
        if st.button("Clear history", disabled=not confirmed):
            with history_connection() as connection:
                connection.execute("DELETE FROM scans")
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Plagiarism Detection Engine", page_icon="🛡️", layout="wide")
    st.markdown(dashboard_css(), unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="eyebrow">TRACEWORD / WRITING INTELLIGENCE</div>'
                '<h1>Bring clarity to every word.</h1>'
                '<p>Explore where your writing meets the web. Find shared passages, follow the evidence, '
                'and review with confidence.</p><span class="hero-tag">One document</span>'
                '<span class="hero-tag">Real web sources</span><span class="hero-tag">Explainable NLP</span></div>',
                unsafe_allow_html=True)
    st.session_state.setdefault("susp_text", "")
    sidebar_settings()
    scan_tab, history_tab, guide_tab = st.tabs(["New scan", "History & insights", "How it works"])
    with history_tab:
        show_history()
    with guide_tab:
        st.subheader("From your words to supporting evidence")
        for title, detail in [("01 / Add your writing", "Upload one document or paste a passage. No reference document needed."),
                              ("02 / Discover sources", "Tavily searches up to ten sampled phrases and returns available source text."),
                              ("03 / Compare and review", "TF-IDF compares word and phrase patterns. Open source links to check context and attribution.")]:
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.write(detail)
        st.info("Overlap is a starting point for review. Quotations, common phrases and citations need context. TF-IDF may miss paraphrases, and web search cannot cover every publication.")
    with scan_tab:
        scan_workspace()


def scan_workspace() -> None:
    st.subheader("What are you working on?")
    st.caption("Give your scan a name so you can find it again in your history.")
    st.text_input("Scan name", key="scan_name", placeholder="e.g. Literature review - first draft", max_chars=100)
    sample_col, clear_col, _ = st.columns([1.4, 0.8, 4])
    sample_col.button(
        "✨ Load sample text", key="btn_samples", on_click=load_samples, width="stretch",
        help="Fill the document with a short sample. Click Check web for matches to search it.",
    )
    clear_col.button("Clear", key="btn_clear", on_click=clear_all, width="stretch")

    editor, companion = st.columns([2.3, 1], gap="large")
    with editor:
        input_pane()
    with companion:
        st.markdown('<div class="workspace-note"><div class="eyebrow">A LITTLE CLARITY</div>'
                    '<h3>Good writing starts with a thoughtful review.</h3>'
                    '<p>Bring your essay, research draft, or article. We’ll help you find shared wording and the sources worth a closer look.</p>'
                    '<div class="step-line"><span class="step-number">1</span>Add your document</div>'
                    '<div class="step-line"><span class="step-number">2</span>Discover public sources</div>'
                    '<div class="step-line"><span class="step-number">3</span>Review the evidence</div></div>', unsafe_allow_html=True)
        text = st.session_state.get("susp_text", "")
        words, phrases = st.columns(2)
        words.metric("Words", f"{len(text.split()):,}")
        phrases.metric("Search phrases", len(build_search_excerpts(text)))
        st.caption("Your scan summary will be saved automatically when the check completes.")
    st.info(
        "When you search, up to 10 short excerpts from this document are sent to Tavily. "
        "The app displays candidate pages; it does not search the entire internet."
    )
    analyze = st.button("🔍 Check web for matches", key="btn_analyze", type="primary", width="stretch")
    if analyze:
        text = st.session_state["susp_text"].strip()
        if not text:
            st.error("Upload a document or paste its text before searching.")
        elif not build_search_excerpts(text):
            st.error("The document needs at least one sentence with eight or more words to search.")
        elif not (api_key := get_tavily_api_key()):
            st.error(
                "Tavily API key is not configured. Set TAVILY_API_KEY as an environment "
                "variable or add it to .streamlit/secrets.toml."
            )
        else:
            progress_bar = st.progress(0, text="Starting your web search...")
            def update_progress(done: int, total: int, message: str) -> None:
                progress_bar.progress(done / max(total, 1), text=message)
            results = None
            with st.spinner("Searching published web sources…"):
                try:
                    results = search_published_sources(text, api_key, progress=update_progress)
                except TavilySearchError as exc:
                    st.error(str(exc))
                    if st.session_state.get("search_results"):
                        st.info("This attempt failed. Any results below are from your previous successful scan.")
                finally:
                    progress_bar.empty()
                if results is not None:
                    results["document_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    st.session_state["search_results"] = results
                    try:
                        save_scan(text, results, st.session_state.get("scan_name", "Untitled scan"),
                                  st.session_state.get("review_threshold", 0.5))
                    except sqlite3.Error:
                        st.warning("Your results are ready, but the history could not be saved.")
                    else:
                        st.rerun()

    search_results = st.session_state.get("search_results")
    if search_results:
        current_hash = hashlib.sha256(st.session_state["susp_text"].strip().encode("utf-8")).hexdigest()
        if search_results["document_hash"] == current_hash:
            show_results(search_results)
        else:
            st.info("The document changed since the last search. Run the web check again for current results.")


if __name__ == "__main__":
    main()
