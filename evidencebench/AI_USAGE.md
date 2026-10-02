# AI Usage & Verification Disclosure

**Project:** EvidenceBench — Retrieval-Augmented Generation & Abstention Gatekeeper  
**Date:** September 2026  
**Document:** `AI_USAGE.md`

---

## 1. Overview of AI Assistance
In compliance with the One-Month Challenge guidelines, AI tools were utilized as interactive pair-programming assistants to accelerate scaffolding, boilerplate generation, and benchmark authoring. All core architectural decisions, mathematical formulations, and critical debugging steps were verified and understood from first principles.

---

## 2. Specific Uses of AI Tools

### 2.1 Code Scaffolding & Domain Modeling
- **Component:** Pydantic domain models in `evidencebench/models.py`.
- **AI Contribution:** Generated initial Pydantic class signatures for `DocumentMetadata`, `Chunk`, `RetrievalCandidate`, `CitationSpan`, `ExecutionTrace`, and `BenchmarkQuestion`.
- **Human Verification:** Reviewed all field constraints, default factories, and enum definitions. Added timezone-aware UTC datetime factories and ensured exact type annotations.

### 2.2 Corpus & Benchmark Generation Scripts
- **Component:** `scripts/generate_corpus.py` and `scripts/generate_benchmark.py`.
- **AI Contribution:** Drafted ReportLab PDF layout scripts and 50 realistic questions across technical specs, financial earnings, climate policy, and consensus protocols.
- **Human Verification:** Manually reviewed and curated all 50 questions to guarantee ground truth accuracy, balanced categorization (Direct, Paraphrased, Multi-Document, Conflicting, Unanswerable), and clear conflicting pairs (e.g. EU 2030 targets 55% vs 40%; Apex EBITDA $950M vs $650M).

### 2.3 Frontend Interface
- **Component:** `evidencebench/web/index.html`.
- **AI Contribution:** Provided Tailwind CSS component layouts for the research workspace dashboard, candidate inspect drawer, and benchmark table.
- **Human Verification:** Verified JavaScript fetch calls, added citation tag click handlers, trace JSON copy functions, and source preview drawer interactivity.

---

## 3. Human Discovery & Resolution of Subtle Algorithmic Bugs

During automated test execution, AI-generated baseline implementations exhibited subtle failure modes that required deep first-principles diagnosis and human engineering fixes:

### Case 1: Naive Regex Sentence Splitting Causing Citation Fragmentation
- **Issue:** The initial sentence splitter used `re.split(r"(?<=[.!?])\s+", text)`.
- **Symptom:** Citation precision dropped to 0.0% because citations like `[Doc: file.pdf, p. 1, § SLA | chunk: chk_001]` were split in half immediately after the abbreviation `p. 1`.
- **Human Resolution:** Diagnosed the abbreviation fragmentation and re-engineered the parser to be bracket-aware, parsing claims on bullet and paragraph boundaries while keeping citation tags unbroken.

### Case 2: PDF Parsing Heading Truncation
- **Issue:** In the initial PDF parser, `end_char` was assigned `start_char + len(line)` inside the line-iteration loop.
- **Symptom:** Chunks generated from PDF documents had lengths of only 28-36 characters (containing only the heading line without body text), causing massive retrieval misses and false abstentions.
- **Human Resolution:** Discovered the offset bug, cleared the cached manifest, and rewrote `_parse_pdf` to scan the assembled text across the full document, expanding chunks from 36 characters to 266-457 characters of complete factual context.

### Case 3: CP1252 Unicode Mismatch on Section Symbol `§`
- **Issue:** Regex patterns containing the raw character `§` failed on Windows environments with default CP1252 locale encoding.
- **Human Resolution:** Replaced raw symbols with Unicode escapes `\u00a7` and added flexible pattern matching allowing `sec`, `section`, or omitted section indicators.

---

## 4. Verification Methodology
All code and experiments were verified through:
1. **Automated Unit & Integration Tests:** 13 pytest tests in `tests/` covering parsing, deduplication, chunking comparison, BM25, dense retrieval, RRF fusion, abstention gates, and citation alignment.
2. **Empirical Ablation Execution:** 5-way benchmark execution script measuring Recall@K, MRR, nDCG, Citation Precision, Citation Coverage, and Abstention F1 over 50 real queries.
3. **Comprehensive Failure Logging:** 22 detailed failures documented in `FAILURE_LOG.md` with explicit diagnostic traces.
