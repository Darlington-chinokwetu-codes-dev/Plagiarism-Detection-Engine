# Plagiarism Detection Engine

Upload or paste one document. Tavily discovers public web sources; local word unigram/bigram TF-IDF cosine similarity compares each document sentence with retrieved source passages.

## Run on Windows

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Create an account at https://app.tavily.com, generate a key, and enter it in the sidebar password field. The key stays in the Streamlit session and is not written to disk. Alternatively set TAVILY_API_KEY in the environment or Streamlit secrets. Never commit credentials.

Click Load sample text or upload TXT, MD, PDF, or DOCX, then Check web for matches. Up to ten sampled phrases leave the app for Tavily. Each request explicitly uses basic search with automatic parameter selection disabled. No AI-generated answer is requested. A scan normally makes up to ten API requests, or up to twenty with retries. Requests may consume credits even if the client times out or a later request fails.

## Slow requests and partial scans

Initial requests ask for page text with a 10-second connection timeout and a 90-second read timeout. A temporary network/timeout/server failure gets one retry after a short delay, without requesting page text. Subsequent searches in that scan also use this lighter mode, with a 45-second read timeout. Search excerpts are labelled as weaker evidence. Authentication, quota and invalid-response errors are not retried.

Progress shows phrases searched. The app stops starting requests after a five-minute soft budget or two consecutive failed phrases. Socket timeouts measure inactivity, so this is not a strict whole-scan time limit. Successful queries are retained in a clearly labelled partial scan if another query fails. All-failed scans produce an error rather than a similarity score. An unsuccessful new attempt keeps previous results, explicitly labelled as previous results.

JSON evidence reports contain `search` metadata (requested/successful query counts, incomplete status, lighter-query count, warnings and threshold), `word_overlap` (percentage, counts, character spans and linked source evidence), and TF-IDF `sentences`. History stores complete/partial status and query counts; partial scans appear as gaps in estimate trends and remain visible in the table. Existing records are migrated without deletion and labelled `legacy`.

## Estimated plagiarism and highlights

The headline **Estimated plagiarism** measures matching wording, not an adjudication of plagiarism. The app tokenizes words with character offsets, case-folds them, and finds sequences of at least five consecutive words in the retrieved source text. Punctuation and case differences are ignored. Overlapping matches across sources count once. The percentage is **matched input words / all input words x 100**. For example, ten matching words in a 100-word document produce 10.0%, regardless of the length of the sentence containing them.

The original document displays matching spans in red. A passage selector shows the matching text alongside the retrieved source excerpt, its evidence type and URL. Downloadable JSON includes all matches and offsets. Quoted and cited wording is still counted and requires human review. Common phrases can match; paraphrases and sources not retrieved can be missed. No usable source text produces **Unable to estimate**; zero means no matching runs were detected in the retrieved content, not proof of originality. The estimate is visibly marked partial when searches failed.

History saves the word-overlap estimate separately from TF-IDF sentence coverage. Older records lacking word-level evidence remain blank in the estimate chart and are never converted from the older metric. Changing the TF-IDF threshold changes the sentence review, not the word-overlap percentage.

## NLP method and interpretation

Sentence segmentation preserves common abbreviations. Lowercased word unigrams and bigrams use sublinear term frequency, inverse document frequency and L2 normalization, fitted jointly to the submitted sentences and retrieved passages. Cosine similarity selects the strongest source passage per sentence. Long source sentences use overlapping 90-word windows with a 60-word stride. Source text is limited to 60,000 characters per page, 10,000 passages overall and 100,000 vocabulary features.

The separate TF-IDF review uses an adjustable default 0.50 threshold, a heuristic that is not benchmark-calibrated. Its sentence coverage is the word-weighted fraction of input sentences of at least eight words above that threshold. This differs from the headline word-overlap estimate. Short sentences are excluded from TF-IDF coverage. Possible cited quotations are marked for review, not automatically cleared. The report includes source passages, URLs, cosine scores and whether evidence came from page text or only a search excerpt. Downloadable JSON contains no API key.

TF-IDF detects shared wording; it does not reliably detect semantic paraphrases. Retrieved sources are incomplete, and search snippets are weaker evidence than page text. No results do not establish originality. Validate thresholds on labeled original, copied, paraphrased and cited examples before making accuracy claims. Legacy semantic helper functions remain in the source but are not part of the active workflow.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Tests use mocked provider responses and require no API key. A live provider check requires your own key.

## Dashboard and history

The Traceword interface has New scan, History & insights, and How it works tabs. Completed scans save summary metrics in a local SQLite database under `.local/` (ignored by Git). Scan names, UTC timestamps, word/source/query counts, review thresholds and coverage are saved; document text and API keys are not. All users of this local app share that history. History includes charts, name filtering, JSON export and a confirmed clear action. Missing evidence is stored as a missing score, not zero. Changing the review threshold later does not rewrite past scans.
