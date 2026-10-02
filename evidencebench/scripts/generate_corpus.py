"""
Generates the test corpus: multi-page PDFs, Markdown documents, and Text files.
Includes realistic domain specs and explicit conflicting documents.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)


def generate_cloud_architecture_pdf():
    pdf_path = CORPUS_DIR / "cloud_architecture_2025.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3a8a"),
    )
    h1_style = ParagraphStyle(
        "H1Style",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e40af"),
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
    )

    story = []

    # Page 1
    story.append(Paragraph("Cloud Architecture Specifications 2025", title_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("1. Executive Summary and SLA Targets", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The NextGen Cloud Platform provides enterprise-grade infrastructure with guaranteed "
        "availability of 99.999% uptime across all Tier-4 availability zones. "
        "The system enforces a strict p99 latency SLA limit of 15ms for internal RPC calls. "
        "In the event of an unplanned primary region outage, the automated disaster recovery protocol "
        "executes cross-region failover within 30 seconds with zero committed data loss.",
        body_style,
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("2. Availability Zone Topology", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Deployments span a minimum of three distinct physical availability zones separated by at least "
        "25 kilometers to protect against local environmental disasters. Inter-zone network traffic travels "
        "over dedicated dark fiber with redundant 400 Gbps optical transport.",
        body_style,
    ))
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("3. Distributed Storage Layer and Consensus", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The distributed object and metadata store relies on the Raft consensus algorithm. "
        "Each cluster maintains a quorum of 5 replicas, requiring a minimum of 3 positive acknowledgments "
        "before committing any write to disk. "
        "Write-ahead logs (WAL) are flushed to non-volatile NVMe storage synchronously before ack. "
        "Snapshot compaction runs automatically every 6 hours or whenever the WAL exceeds 10,000 log entries.",
        body_style,
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("4. Storage Durability and Replication", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Object data achieves an annual durability guarantee of eleven nines (99.999999999%). "
        "Data chunks are erasure coded using an 8+4 Reed-Solomon scheme across failure domains. "
        "Storage tiering transitions cold objects to deep archive storage after 90 days of inactivity.",
        body_style,
    ))
    story.append(PageBreak())

    # Page 3
    story.append(Paragraph("5. Security, Encryption, and Egress Pricing", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "All data at rest is encrypted using AES-256-GCM with customer-managed keys rotated annually. "
        "In-transit communication requires mandatory TLS 1.3 encryption with forward secrecy. "
        "Tenant API access is strictly throttled to a default rate limit of 10,000 req/sec per tenant. "
        "Standard external data egress pricing is fixed at $0.05 per gigabyte for the first 50 terabytes, "
        "discounted to $0.03 per gigabyte thereafter.",
        body_style,
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("6. Audit Logging and Compliance", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Immutable audit trails capture all administrative operations and access requests. "
        "Audit logs are retained for 7 years in write-once-read-many (WORM) compliant storage "
        "fulfilling SOC-2 Type II, ISO 27001, and HIPAA compliance mandates.",
        body_style,
    ))

    doc.build(story)
    print(f"Generated {pdf_path}")


def generate_financial_pdf():
    pdf_path = CORPUS_DIR / "financial_q3_report_alpha.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#065f46"),
    )
    h1_style = ParagraphStyle(
        "H1Style",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#047857"),
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
    )

    story = []

    # Page 1
    story.append(Paragraph("Apex Financial Corp - Q3 Official Earnings Report", title_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("1. Consolidated Financial Highlights", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Apex Financial Corp reported total quarterly revenue of $4.2B for Q3 2024, representing "
        "an increase of 14% year-over-year. GAAP net income reached $820M. "
        "The consolidated gross margin was 68%, driven by cloud software license expansion. "
        "The company reported official adjusted EBITDA of $950M with an EBITDA margin of 22.6%.",
        body_style,
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("2. Cash Flow and Liquidity", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Operating cash flow reached $1.1B during the quarter. Cash and cash equivalents totaled "
        "$3.4B at the close of Q3. Free cash flow conversion stood at 85% of adjusted EBITDA.",
        body_style,
    ))
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("3. Capital Expenditures and R&D Investment", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Total R&D investment for the quarter was $640M, focused primarily on generative AI models "
        "and distributed database engineering. Capital expenditures totaled $310M, representing "
        "infrastructure buildouts in Frankfurt and Tokyo.",
        body_style,
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("4. Full-Year 2024 Guidance", h1_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Management updated full-year revenue guidance to a range between $16.8B and $17.1B. "
        "Full-year operating margin is anticipated to remain between 21% and 23%.",
        body_style,
    ))

    doc.build(story)
    print(f"Generated {pdf_path}")


def generate_markdown_docs():
    # 1. Climate Policy 2024
    p1 = CORPUS_DIR / "climate_policy_eu_2024.md"
    p1.write_text(
        """# EU Climate Policy Framework 2024

## 1. Emissions Reduction Targets
The European Union has codified a legally binding greenhouse gas emissions reduction target of 55% by 2030 compared to 1990 baseline levels. This target is anchored in the European Climate Law and forms the cornerstone of the Fit for 55 regulatory package. Net-zero emissions must be achieved across all member states no later than 2050.

## 2. Renewable Energy Directive
Member states are mandated to achieve a minimum 42.5% share of renewable energy in gross final energy consumption by 2030, with an aspirational target of reaching 45%. Permitting processes for wind and solar installations have been streamlined under designated acceleration zones.

## 3. Just Transition Mechanism and Budget
The Just Transition Fund has allocated a total budget of 19.2 billion EUR to assist carbon-intensive regions in retraining industrial workers and modernizing energy infrastructure. Coal phase-out deadlines remain fixed for 2038 across participating member states.
""",
        encoding="utf-8",
    )
    print(f"Generated {p1}")

    # 2. Conflicting Climate Policy 2025 Revision
    p2 = CORPUS_DIR / "climate_policy_eu_revision_2025.md"
    p2.write_text(
        """# EU Climate Policy Framework 2025 Revision

## 1. Emissions Reduction Targets Amendment
Under the 2025 revised parliamentary directive, the 2030 net greenhouse gas emissions target has been revised to 40% reduction due to industrial supply chain bottlenecks and macroeconomic adjustments. The earlier 55% target was officially superseded by Resolution 2025/B9.

## 2. Just Transition Fund Budget Reallocation
The revised 2025 budget for the Just Transition Mechanism is set to 28.5 billion EUR, reflecting an additional emergency modernization tranche. National allocations have been adjusted to prioritize heavy chemical sectors.
""",
        encoding="utf-8",
    )
    print(f"Generated {p2}")

    # 3. Conflicting Financial Report
    p3 = CORPUS_DIR / "financial_q3_report_beta_discrepancy.md"
    p3.write_text(
        """# Equity Research Note: Apex Financial Q3 Audit Review

## 1. Disputed Operating Metrics
Analyst Beta published an independent audit note disputing Apex Financial's Q3 performance figures. The research note indicates that adjusted EBITDA was $650M with an EBITDA margin of 15.5%, contradicting the company's official filing of $950M due to capital lease capitalization disputes.

## 2. Valuation and Rating
Due to conflicting EBITDA calculations and deferred revenue treatments, the stock rating was downgraded from Overweight to Neutral, with a price target reduced from $140 to $105.
""",
        encoding="utf-8",
    )
    print(f"Generated {p3}")


def generate_text_docs():
    # 1. Distributed Systems Consensus
    p1 = CORPUS_DIR / "distributed_systems_consensus.txt"
    p1.write_text(
        """=== Consensus Mechanisms in Distributed Computing ===

Raft is a consensus algorithm designed for understandability and operational safety. A Raft cluster consists of a leader, followers, and candidates. The leader maintains heartbeat intervals of 150ms to prevent follower election timeouts.

When a follower does not receive a heartbeat within the election timeout (randomized between 150ms and 300ms), it transitions to candidate state, increments its term counter, and requests votes from peers.

Log entries are committed only when safely replicated across a majority quorum. Raft guarantees the Leader Completeness Property: if a log entry is committed in a given term, that entry will be present in the logs of the leaders for all higher-numbered terms.

=== Paxos Comparison ===
Multi-Paxos provides equivalent consensus guarantees but merges phase 1 rounds during steady-state leader operation. However, implementation complexity is significantly higher than Raft, particularly regarding log hole filling and membership reconfiguration.
""",
        encoding="utf-8",
    )
    print(f"Generated {p1}")

    # 2. Medical Device ISO Compliance
    p2 = CORPUS_DIR / "medical_device_iso_compliance.txt"
    p2.write_text(
        """=== Medical Device Quality Management ISO 13485 ===

ISO 13485:2016 specifies requirements for a comprehensive quality management system (QMS) where an organization must demonstrate its ability to provide medical devices and related services that consistently meet customer and applicable regulatory requirements.

Design controls require formal Design History Files (DHF) documenting design inputs, outputs, verification, validation, and design transfer. Risk management must adhere strictly to ISO 14971 principles throughout the entire product life cycle.

=== Post-Market Surveillance ===
Manufacturers must establish documented procedures to gather, evaluate, and report post-production customer complaints, adverse events, and vigilance reporting within statutory deadlines (e.g., 15 days for serious public health threats under EU MDR).
""",
        encoding="utf-8",
    )
    print(f"Generated {p2}")


if __name__ == "__main__":
    generate_cloud_architecture_pdf()
    generate_financial_pdf()
    generate_markdown_docs()
    generate_text_docs()
    print("All test corpus files generated successfully.")
