"""
Temporary read-only retrieval inspection for golden dataset Step 4.
Uses existing NEXUS store + MMR retriever only. No LLM, no DB writes.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rag.vectorstore.store import get_or_create_store
from rag.vectorstore.retriever import get_mmr_retriever

QUESTIONS: dict[str, str] = {
    "Q2": (
        "What is the exact count of total trainable parameters in the RAG models "
        "described by Patrick Lewis et al., and how are these parameters distributed "
        "among its sub-components?"
    ),
    "Q5": (
        "What are the mathematical formula and the specific learning rate warmup "
        "steps value used to configure the learning rate scheduler in the Transformer "
        "model training?"
    ),
    "Q8": (
        "Synthesize a comparative analysis of the computational environments, hardware "
        "(GPU models and quantity), and training runtimes utilized to train the "
        "Transformer (base vs. big) and the RAG model configurations."
    ),
    "Q9": (
        "How do the high-level policy governance requirements for third-party supply "
        "chain risk (C-SCRM) in the NIST CSF 2.0 compare to the specific technical "
        "vulnerabilities addressed by OWASP Top Ten 2021 Category A08 (Software & "
        "Data Integrity Failures)?"
    ),
    "Q10": (
        "Based on RAG's temporal evaluation experiments, how did swapping Wikipedia "
        "index versions (December 2016 vs. December 2018) impact accuracy when querying "
        "about world leaders whose positions had changed between those dates?"
    ),
    "Q11": (
        "Detail the transitions and merges of OWASP Top Ten security categories from "
        "the 2017 edition to the 2021 edition. Specifically, identify which three 2017 "
        "categories were subsumed into other classifications."
    ),
    "Q12": (
        "Summarize the English-to-German and English-to-French WMT 2014 translation "
        "test results (BLEU scores) for both the Transformer base and big "
        "configurations, and identify which previous state-of-the-art models they "
        "outperformed."
    ),
    "Q14": (
        "Contrast the mathematical and architectural mechanisms by which RAG-Sequence "
        "and RAG-Token marginalize over latent retrieved documents during sequence "
        "generation."
    ),
    "Q15": (
        "Differentiate between the root causes targeted by OWASP Top Ten Category A04 "
        "(Insecure Design) and Category A08 (Software and Data Integrity Failures), "
        "explaining how their lifecycle positions and threat vectors differ."
    ),
    "Q16": (
        "Compare the computational complexity per layer, minimum number of sequential "
        "operations, and maximum path lengths of Self-Attention layers against Recurrent "
        "and Convolutional layer types."
    ),
    "Q17": (
        'Differentiate between the structural definition and operational purpose of '
        '"CSF Organizational Profiles" and "CSF Tiers" as defined in the NIST CSF 2.0.'
    ),
    "Q18": (
        "Contrast RAG's dense retriever (DPR) with a traditional word overlap-based "
        "BM25 retriever based on task evaluation performance, and identify which "
        "specific task BM25 outperforms DPR in and why."
    ),
    "Q19": (
        'How does the NIST CSF 2.0 describe the conceptual relationship, overlap, and '
        'practical distinctions between "Cybersecurity Risk" and "Privacy Risk"?'
    ),
    "Q20": (
        "What are the specific security configurations, operating system parameters, and "
        "prescriptive access control list (ACL) rules mandated by the NIST CSF 2.0 to "
        "protect endpoints from ransomware?"
    ),
    "Q21": (
        "What are the specific mathematical parameters and weight training updates "
        "applied to RAG's Wikipedia document encoder (BERT_d) during retriever "
        "fine-tuning?"
    ),
    "Q23": (
        "Provide the detailed, step-by-step secure coding checklist and unit testing "
        "suite recommended by OWASP to prevent Server-Side Request Forgery (SSRF) "
        "vulnerabilities."
    ),
    "Q24": (
        'Describe the mathematical formulas and loss parameters used to configure the '
        '"Null Document" probability mechanism in the final evaluated RAG-Sequence model.'
    ),
    "Q25": (
        "Provide the prescriptive step-by-step compliance auditing checklist and timeline "
        "requirements an organization must execute to achieve a certified NIST CSF 2.0 "
        "Tier 4 (Adaptive) maturity rating."
    ),
}


def format_chunk(index: int, doc) -> str:
    meta = doc.metadata or {}
    source = meta.get("source", "unknown")
    page = meta.get("page", "unknown")
    chunk_id = meta.get("chunk_id", "unknown")
    return (
        f"### Retrieved Chunk {index}\n\n"
        f"Source: {source}\n"
        f"Page: {page}\n"
        f"Chunk ID: {chunk_id}\n\n"
        f"```text\n{doc.page_content}\n```\n"
    )


def main() -> None:
    store = get_or_create_store()
    retriever = get_mmr_retriever(store)

    lines: list[str] = [
        "# NEXUS Retrieval Inspection — Step 4",
        "",
        "Read-only inspection using production `get_or_create_store()` + "
        "`get_mmr_retriever()`. No LLM invoked.",
        "",
        f"Collection: `{store._collection.name}` ({store._collection.count()} chunks)",
        "",
    ]

    for qid, question in QUESTIONS.items():
        docs = retriever.invoke(question)
        lines.append(f"## QUESTION {qid}")
        lines.append("")
        lines.append("### Question")
        lines.append(question)
        lines.append("")

        if not docs:
            lines.append("_No chunks retrieved._")
            lines.append("")
            continue

        for i, doc in enumerate(docs, start=1):
            lines.append(format_chunk(i, doc))

    output_path = project_root / "retrieval_inspection.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path} ({len(QUESTIONS)} questions)")


if __name__ == "__main__":
    main()
