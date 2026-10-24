# Plagiarism Detection Engine

An NLP-based plagiarism detection system designed to identify **copied and paraphrased content** by comparing submitted text or documents against reference sources.

The system combines **Natural Language Processing, TF-IDF, cosine similarity, and semantic sentence embeddings** to provide an interpretable plagiarism assessment rather than relying only on exact word matching.

🌐 **Live Application:**  
https://plagiarism-detection-engine-nlp-project.streamlit.app/

---

## About the Project

Traditional plagiarism checking based only on exact keyword matching can struggle when text has been paraphrased, reordered, or expressed using different words.

The **Plagiarism Detection Engine** was developed to provide a more flexible document-comparison workflow.

Users can:

- Upload a document or paste text directly
- Extract and preprocess document content
- Compare the submission with reference documents
- Detect strong lexical copying
- Detect semantically similar or paraphrased text
- View matching passages
- View copied and paraphrased percentages
- Receive an overall plagiarism percentage
- Receive an interpretable risk level
- Generate downloadable reports

The system is intended to **support academic review**, not replace human judgement.

---

## Live Demo

Try the deployed Streamlit application:

**https://plagiarism-detection-engine-nlp-project.streamlit.app/**

The web interface allows users to submit text or documents, run plagiarism analysis, inspect matched passages, and review the generated plagiarism report.

---

## Main Features

### Document Upload and Text Input

The system supports both:

- File upload
- Direct copy-and-paste text input

Supported document formats include:

- `.txt`
- `.md`
- `.pdf`
- `.docx`

---

### NLP Preprocessing

Before comparison, submitted text is processed using NLP techniques such as:

- Unicode normalization
- Sentence segmentation
- Tokenization
- Stop-word removal
- Lemmatization
- Text normalization
- Removal or exclusion of references where configured
- Exclusion of quoted material where configured

The project uses **NLTK** for core NLP operations such as sentence tokenization, English stop words, and WordNet lemmatization.

---

## TF-IDF Feature Extraction

Eligible sentences are transformed into numerical representations using **TF-IDF**.

The implementation uses lexical features including:

- Unigrams
- Bigrams
- TF-IDF weighting
- Cosine similarity

This makes it possible to identify sentences that share significant wording even when the entire document is not identical.

---

## Semantic Similarity

Lexical matching alone may miss plagiarism when a sentence has been heavily paraphrased.

To improve detection, the system can use semantic sentence embeddings through:

`sentence-transformers/all-MiniLM-L6-v2`

Semantic similarity helps identify sentences that express similar meaning even when their exact words differ.

---

## How It Works

```text
User Document / Pasted Text
            |
            v
     Text Extraction
            |
            v
     NLP Preprocessing
            |
            v
   Sentence Segmentation
            |
            v
   TF-IDF Representation
            |
            +-------------------+
            |                   |
            v                   v
 Lexical Similarity      Semantic Similarity
            |                   |
            +---------+---------+
                      |
                      v
            Similarity Analysis
                      |
                      v
          Copy / Paraphrase Label
                      |
                      v
          Plagiarism Percentage
                      |
                      v
             Risk Classification
                      |
                      v
       Matched Passages + Report
```

---

## Detection Logic

The plagiarism engine compares submitted sentences with sentences from reference documents.

The default configuration uses thresholds such as:

| Setting | Default |
|---|---:|
| Minimum sentence length | 4 tokens |
| Lexical similarity threshold | 0.60 |
| Copy threshold | 0.85 |
| Semantic similarity threshold | 0.80 |
| Shingle size | 4 words |

A strong lexical match can be classified as a **copy**, while semantic similarity can help identify **paraphrased content**.

These values are configurable and can be tuned using evaluation results.

---

## Risk Levels

The final plagiarism percentage is converted into an easy-to-understand risk category.

| Plagiarism Percentage | Risk Level |
|---:|---|
| Below 10% | Minimal |
| 10% – below 30% | Low |
| 30% – below 60% | Moderate |
| 60% and above | High |

The risk level is intended as a screening aid and should always be interpreted together with the matched evidence.

---

## Results Provided

After analysis, the system can display:

- Overall plagiarism percentage
- Copied percentage
- Paraphrased percentage
- Risk level
- Number of analysed sentences
- Matching reference document
- Similarity score for each match
- Matched source passages
- Matched submitted passages
- Source-level coverage
- Lexical similarity information
- Semantic similarity information

This evidence makes the result easier to review than a plagiarism percentage alone.

---

## Report Generation

Results can be exported into structured reports.

Supported report outputs include:

- HTML
- JSON

The generated reports preserve the plagiarism score, risk level, source information, and matched passages for later review.

---

## Published Source Search

The deployed version can also use an external search service for published-source discovery when the required API key is configured.

API keys should **never be committed directly to GitHub**.

For Streamlit deployment, store secrets securely through **Streamlit Secrets**.

Example:

```toml
TAVILY_API_KEY = "your_api_key_here"
```

Do not place the real API key inside source files or this README.

---

## Technology Stack

### Language

- Python

### NLP and Machine Learning

- NLTK
- scikit-learn
- TF-IDF
- Cosine Similarity
- Sentence Transformers
- Regular Expressions

### Web Application

- Streamlit

### Backend / API

- FastAPI
- Uvicorn

### Document Processing

- PDF text extraction
- DOCX text extraction
- Plain-text processing
- Markdown processing

### Reporting

- HTML
- JSON

---

## Project Architecture

```text
Plagiarism Detection Engine
│
├── User Interface
│   └── Streamlit Web Application
│
├── Document Extraction
│   ├── TXT
│   ├── Markdown
│   ├── PDF
│   └── DOCX
│
├── NLP Preprocessing
│   ├── Sentence Tokenization
│   ├── Stop-word Removal
│   ├── Lemmatization
│   └── Normalization
│
├── Feature Extraction
│   ├── TF-IDF
│   └── Sentence Embeddings
│
├── Similarity Engine
│   ├── Lexical Similarity
│   ├── Semantic Similarity
│   └── Shingle Containment
│
├── Classification
│   ├── Copy
│   ├── Paraphrase
│   └── No Significant Match
│
├── Risk Analysis
│   ├── Minimal
│   ├── Low
│   ├── Moderate
│   └── High
│
└── Reporting
    ├── HTML
    └── JSON
```

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-FOLDER>
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

Run:

```bash
streamlit run app.py
```

Then open the local address displayed by Streamlit, normally:

```text
http://localhost:8501
```

---

## Running the FastAPI Backend

The project also includes a FastAPI service.

Start it with:

```bash
uvicorn api.main:app --reload
```

The API should then be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Capabilities

The backend supports operations such as:

- Checking service health
- Viewing the reference library
- Adding reference documents
- Removing reference documents
- Comparing submitted text/documents
- Generating plagiarism reports

The reference library can therefore be updated without rebuilding the plagiarism engine itself.

---

## Configuration

Some detection settings can be tuned from the project configuration.

Examples include:

```python
lexical_threshold = 0.60
copy_threshold = 0.85
semantic_threshold = 0.80

risk_low = 10.0
risk_moderate = 30.0
risk_high = 60.0
```

The semantic engine can also be configured through the environment.

Example:

```bash
PDE_SEMANTIC=auto
```

Possible modes include:

```text
auto
none
sbert
```

A custom reference-library directory can also be configured through:

```bash
PDE_LIBRARY_DIR=data/library
```

---

## Example Output

A completed plagiarism check can produce information similar to:

```text
Overall Plagiarism : 48.5%
Copied Content     : 31.2%
Paraphrased Content: 17.3%
Risk Level         : Moderate
```

The result also includes the individual passages responsible for the plagiarism score, allowing the reviewer to inspect the evidence directly.

---

## Evaluation

The project includes controlled test data for implementation validation.

These tests help verify whether the engine can distinguish between:

- Strong copied content
- Paraphrased content
- Low-similarity or unrelated text

The evaluation should not be interpreted as a large-scale real-world benchmark. A larger labelled plagiarism corpus would be required for rigorous production-level evaluation.

---

## Why This Project Is Different

Instead of returning only one unexplained similarity number, the system attempts to show **why** content was flagged.

It combines:

**Lexical similarity**

for identifying strong word-level overlap,

with:

**Semantic similarity**

for identifying paraphrased meaning.

The user can then inspect the matching passages and make the final academic decision.

---

## Limitations

Like any automated plagiarism detector, this project has limitations.

- Semantic similarity does not automatically prove plagiarism.
- Common academic phrases may naturally appear in multiple documents.
- Very short sentences may not contain enough information for reliable comparison.
- Highly creative paraphrasing can still be difficult to detect.
- Citations and legitimate quotations require human interpretation.
- Results depend on the quality and size of the reference-document collection.
- Online-source discovery depends on the availability of external search services.

For these reasons, the application should be used as a **decision-support tool rather than an automatic academic misconduct judge**.

---

## Future Improvements

Possible future enhancements include:

- Larger labelled plagiarism evaluation datasets
- Stronger transformer-based semantic models
- Citation-aware plagiarism detection
- Improved quotation handling
- Cross-language plagiarism detection
- Multilingual NLP support
- Source ranking
- More detailed analytics dashboards
- Improved explainability
- Batch document processing
- Larger institutional reference libraries
- Better web-source discovery
- User authentication and role-based access
- Persistent plagiarism history
- Database integration

---

## Project Team

### Trinity E Chigubhu
**Roll Number:** 24B11CS536

### Blessed Zambezi
**Roll Number:** 24B11CS537

### Darlington Chinokwetu
**Roll Number:** 24B11CS538

---

## Academic Information

**Project:** Plagiarism Detection Engine  
**Domain:** Natural Language Processing  
**Programme:** B.Tech — Computer Science and Engineering  
**Academic Year:** 2026–2027  
**Institution:** Aditya University, Surampalem

### Project Guide

**Mr. Talluri Haribabu**  
Assistant Professor  
Department of Computer Science and Engineering

---

## Responsible Use

This project is intended for:

- Academic learning
- NLP experimentation
- Similarity analysis
- Plagiarism screening
- Research and educational demonstrations

A plagiarism score should **not be treated as final proof of academic misconduct**.

Human review of citations, quotations, context, matched passages, and source attribution should always form part of the final decision.

---

## License

This project was developed as an academic project.

If you intend to reuse, modify, or distribute the code, add the appropriate license to the repository.

---

## Feedback

Suggestions and improvements are welcome.

If you find a bug or have an idea for improving the plagiarism detection process, feel free to open an issue or contribute to the project.

---

**Built with Python, NLP and a focus on explainable plagiarism detection.**
