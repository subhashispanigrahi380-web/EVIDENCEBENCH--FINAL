# EvidenceBench: 5–8 Minute Demo Video Script & Walkthrough

This document provides a timed, step-by-step presentation script and test vector guide for recording the 5–8 minute demonstration video.

---

## Video Outline & Timing Breakdown

| Timecode | Segment | Core Focus | Visual Display |
| :---: | :--- | :--- | :--- |
| **0:00 – 1:00** | Introduction & Problem Space | Why standard RAG chat boxes fail; need for verifiable evidence and abstention | Slide / Architecture Diagram |
| **1:00 – 2:30** | System Architecture Overview | Ingestion, Structure-Aware Chunking, Dual-Channel BM25 + Dense, Calibrated RRF | Mermaid Architecture Diagram & Code |
| **2:30 – 4:00** | Live Demo: Grounded Citations | Direct Query & Paraphrased Query with Clickable Passage Preview Drawer | EvidenceBench Web Workspace UI |
| **4:00 – 5:30** | Live Demo: Abstention Gatekeeper | Conflicting Evidence Refusal & Out-of-Domain Query Refusal | EvidenceBench Web Workspace UI |
| **5:30 – 7:00** | Benchmark & Ablation Study | 50-Question Benchmark, Recall@5 (100%), Citation Precision (100%), Failure Inspector | Benchmark Tab in Web UI |
| **7:00 – 8:00** | Codebase Walkthrough & Conclusion | Test Suite (100% pass), Failure Log, CLI commands, Summary | Terminal / VS Code / README |

---

## Step-by-Step Narration Script

### [0:00 – 1:00] Introduction & Problem Space
**Presenter:**
> *"Welcome to the demonstration of EvidenceBench. In traditional RAG applications, the chat box is the easy part. The real difficulty lies in parsing, chunking, hybrid retrieval, cross-encoder reranking, citation alignment, and knowing when to abstain.
> Standard RAG systems will confidently invent answers even when the corpus is contradictory, weak, or completely silent on the topic.
> EvidenceBench is an evidence-first knowledge system that ingests PDFs, Markdown, and text, attributes every claim to specific passage-level citations, and enforces an Abstention Gatekeeper that refuses to answer when evidence is weak, missing, or conflicting."*

---

### [1:00 – 2:30] Architecture Walkthrough
**Presenter:**
> *"Here is the EvidenceBench architecture:
> First, documents are ingested through a unified parser that computes SHA-256 fingerprints to eliminate duplicate vectorization while tracking document version lineage.
> Second, we implement two chunking strategies: Fixed-Recursive and Structure-Aware. The Structure-Aware chunker preserves complete section hierarchies and prepends metadata breadcrumbs—like document name, section title, and page number—directly into each chunk representation.
> Third, retrieval combines two distinct channels: first-principles Okapi BM25 keyword matching and all-MiniLM dense semantic embeddings.
> Fourth, we fuse them using our own Calibrated Reciprocal Rank Fusion (C-RRF), which combines rank stability with min-max normalized score intensity before passing the candidate pool to a Cross-Encoder reranker.
> Finally, our Abstention Gatekeeper intercepts the candidates, checking for missing concepts, low confidence, or contradictory numbers between documents before any answer is synthesized."*

---

### [2:30 – 4:00] Live Demo: Grounded Citations & Source Viewer
**Action:** Open Web UI at `http://localhost:8000`.

**Query 1 (Direct Factual):**
> *"What is the guaranteed uptime availability in Cloud Architecture Specifications 2025?"*

**Presenter:**
> *"Let's test a direct factual query. Notice that the system returns an answer with a green badge: ANSWERED with 0.758 confidence.
> Every factual statement is tagged with a passage-level citation: [Doc: cloud_architecture_2025.pdf, p. 1, § 1. Executive Summary and SLA Targets | chunk: chk_struct_cloud__000].
> When I click on this citation tag, the right-hand Source Preview Drawer opens instantly, highlighting the exact passage, document filename, page number, and section breadcrumbs.
> Citation precision is calculated at 100%, and coverage is 100%."*

**Query 2 (Paraphrased Conceptual):**
> *"How resilient is the cloud infrastructure against service interruptions and downtime?"*

**Presenter:**
> *"Next, let's submit a paraphrased query that shares almost no exact vocabulary with the source text.
> Notice how BM25 alone would struggle here, but our dense embedding channel retrieves the availability zone failover passage at rank #1 with high cosine similarity, and the cross-encoder confirms relevance."*

---

### [4:00 – 5:30] Live Demo: The Abstention Gatekeeper in Action

**Query 3 (Conflicting Evidence):**
> *"What is the official EU 2030 greenhouse gas emissions reduction target?"*

**Presenter:**
> *"Now let's test a critical capability: knowing when to abstain.
> In our corpus, the 2024 Climate Policy document establishes a 55% reduction target, while the 2025 Revised Framework amends this target to 40%.
> Instead of hallucinating or arbitrarily choosing one number, EvidenceBench triggers the Abstention Gatekeeper.
> The badge turns red: [SYSTEM REFUSAL - CONFLICTING_EVIDENCE].
> The system explicitly states: 'Refusing to answer due to contradictory sources: Conflicting metric numbers reported for reduction between climate_policy_eu_2024.md (55%) and climate_policy_eu_revision_2025.md (40%).'
> This protects users from making decisions on contradictory data."*

**Query 4 (Unanswerable / Out-of-Domain):**
> *"What was the retail release price of the Apple iPhone 16 Pro in Tokyo?"*

**Presenter:**
> *"Now we query an out-of-domain topic completely absent from our corpus.
> The gatekeeper recognizes that query entities have 0.0% overlap with the retrieved candidates and refuses to answer: [SYSTEM REFUSAL - MISSING_EVIDENCE]. Zero hallucinations."*

---

### [5:30 – 7:00] Benchmark Suite & Ablation Dashboard
**Action:** Click the "Benchmark" tab in the Web UI.

**Presenter:**
> *"Under the Benchmark tab, we can run our complete 50-question curated test suite covering Direct, Paraphrased, Multi-Document, Conflicting, and Unanswerable queries.
> In our ablation study comparing 5 configurations:
> - Dense-Only achieved 100% Recall@5 and 81.2% Recall@1.
> - BM25-Only achieved 90.6% Recall@5.
> - Our Calibrated Hybrid Fusion achieved the highest Abstention F1 score of 77.8% and 84.0% overall accuracy.
> - Across all configurations, Citation Precision maintained 100.0%.
> Below the metrics grid, our Failure Inspector logs and categorizes every miss, helping engineers diagnose retrieval misses and tuning thresholds."*

---

### [7:00 – 8:00] Codebase Walkthrough & Conclusion
**Action:** Show Terminal with `pytest` execution.

**Presenter:**
> *"All 13 automated tests pass cleanly with pytest.
> The repository includes complete documentation:
> - ARCHITECTURE.md detailing the mathematical formulations of Calibrated RRF and abstention gating.
> - EXPERIMENT_REPORT.md containing full ablation tables.
> - FAILURE_LOG.md documenting 22 real failure cases and root causes.
> - AI_USAGE.md disclosing all AI assistance and verification steps.
> Thank you for exploring EvidenceBench."*
