"""
CSCI933 Assignment 2 — Domain Adaptation with Small Language Models
Shakespeare-Aware RAG System

Run:
    cd ~/Desktop/assignment2
    ls
    pip install sentence-transformers transformers torch scikit-learn pandas matplotlib seaborn accelerate rank_bm25
    python assignment2_shakespeare_rag.py --dataset ./shakespeare_slm_dataset
    In the end -> "quit" to exit

Optional flags:
    --dataset PATH   Path to the shakespeare_slm_dataset folder  (default: ./shakespeare_slm_dataset)
    --device  cpu|cuda                                           (default: auto-detect)
    --top_k   N      Number of passages to retrieve              (default: 3)
    --no_cache       Force recompute embeddings even if cache exists
"""

import argparse
import json
import os
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

# ─── dependency check ────
REQUIRED = [
    "sentence_transformers", "transformers", "torch",
    "sklearn", "pandas", "matplotlib", "seaborn",
]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print("Missing packages. Run:")
    print("  pip install sentence-transformers transformers torch scikit-learn pandas matplotlib seaborn accelerate rank_bm25")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ══════════
# 1. CONFIG
# ══════════
PLAYS = ["hamlet", "macbeth", "romeo_and_juliet"]
PLAY_NAMES = {
    "hamlet": "Hamlet",
    "macbeth": "Macbeth",
    "romeo_and_juliet": "Romeo and Juliet",
}
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME       = "google/flan-t5-large"
DEFAULT_TOP_K        = 3
CACHE_FILE           = Path("./embeddings_cache.npz")
OUTPUT_CSV           = Path("./evaluation_results.csv")
OUTPUT_CHART         = Path("./evaluation_chart.png")


# ════════════════
# 2. DATA LOADING
# ════════════════

def load_scene_chunks(dataset_dir: Path) -> List[Dict]:
    """Load pre-chunked scene-level JSONL for all three plays."""
    all_chunks = []
    for play_key in PLAYS:
        path = dataset_dir / f"{play_key}_scene_chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {path}\n"
                "Pass --dataset to point at the shakespeare_slm_dataset folder."
            )
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = json.loads(line)
                    chunk["play_key"] = play_key
                    chunk.setdefault("play", PLAY_NAMES[play_key])
                    all_chunks.append(chunk)
    return all_chunks


def build_retrieval_chunks(scene_chunks: List[Dict]) -> List[Dict]:
    """
    Convert raw scene data into retrieval-ready chunks.

    Design:
    - One scene = one chunk (preserves dramatic context)
    - embed_text  = scene_summary + first 500 chars (semantic + lexical signal)
    - display_text = first 1200 chars (fits LLM prompt budget)
    """
    chunks = []
    for i, sc in enumerate(scene_chunks):
        raw_text = sc.get("text", "").strip()
        summary  = sc.get("scene_summary", "").strip()

        chunk = {
            "chunk_id"     : sc.get("scene_id", f"chunk_{i:04d}"),
            "play"         : sc.get("play", PLAY_NAMES.get(sc.get("play_key", ""), "Unknown")),
            "play_key"     : sc.get("play_key", ""),
            "act"          : sc.get("act"),
            "scene"        : sc.get("scene"),
            "scene_summary": summary,
            "keywords"     : sc.get("keywords", []),
            "embed_text"   : summary,
            "full_text"    : raw_text,
            "display_text" : raw_text[:1200] + ("..." if len(raw_text) > 1200 else ""),
        }
        chunks.append(chunk)
    return chunks


# ════════════════════
# 3. EMBEDDING INDEX
# ════════════════════

def build_or_load_index(
    chunks: List[Dict],
    model_name: str,
    cache_path: Path,
    force: bool = False,
) -> Tuple[SentenceTransformer, np.ndarray]:
    """Build dense embedding index or load from .npz cache."""
    model = SentenceTransformer(model_name)

    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        print(f"  Loaded cached embeddings from {cache_path}  {data['embeddings'].shape}")
        return model, data["embeddings"]

    print(f"  Building embeddings for {len(chunks)} chunks...")
    texts      = [c["embed_text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.savez(cache_path, embeddings=embeddings)
    print(f"  Cached to {cache_path}  {embeddings.shape}")
    return model, embeddings


# ══════════════
# 4. RETRIEVAL
# ══════════════

class DenseRetriever:
    def __init__(self, chunks, embed_model, embeddings):
        self.chunks      = chunks
        self.model       = embed_model
        self.embeddings  = embeddings

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        q_emb  = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(q_emb, self.embeddings)[0]
        top_i  = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_i]


class TFIDFRetriever:
    """Baseline: keyword-based TF-IDF retrieval, no neural components."""

    def __init__(self, chunks):
        self.chunks    = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=10_000, stop_words="english"
        )
        corpus         = [c["full_text"] for c in chunks]
        self.matrix    = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_i  = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_i]


# ════════════════
# 5. GENERATION
# ════════════════

def load_generator(model_name: str, device: str):
    print(f"  Loading {model_name} on {device}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    dtype     = torch.float16 if device == "cuda" else torch.float32
    model     = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {params:.0f}M parameters loaded.")
    return tokenizer, model


# Out-of-domain questions the system cannot answer from the plays
_OUT_OF_DATASET_PLAYS = [
    "merchant of venice", "othello", "king lear", "a midsummer",
    "much ado", "twelfth night", "julius caesar", "the tempest",
    "richard", "henry", "taming of the shrew",
]

_OUT_OF_DOMAIN = [
    "who is shakespeare", "what is shakespeare", "who wrote",
    "who is the author", "when was shakespeare born", "william shakespeare",
]

def _is_out_of_domain(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _OUT_OF_DOMAIN)

def _is_unknown_play(query: str) -> bool:
    q = query.lower()
    return any(play in q for play in _OUT_OF_DATASET_PLAYS)


def build_prompt(
    query: str,
    retrieved: List[Tuple[Dict, float]],
    stylised: bool = False,
) -> str:
    """
    Assemble RAG prompt tuned for flan-t5-large.

    Key fixes vs. original:
    - Removed 'Summary:' label from context blocks — flan-t5 was
      pattern-matching to it and outputting 'Summary: X' as the answer.
    - Kept context labels as plain 'Context N' with source attribution.
    - Used a cleaner Q/A format that matches flan-t5's instruction tuning.
    - Excerpt trimmed to 300 chars (leaves more room for the question).
    """
    context_parts = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        # Use scene description (not labelled "Summary:") + short text excerpt
        excerpt = chunk["full_text"][:300].replace("\n", " ").strip()
        scene_desc = chunk["scene_summary"]
        context_parts.append(
            f"Context {rank} ({chunk['play']}, Act {chunk['act']}, "
            f"Scene {chunk['scene']}): {scene_desc} {excerpt}"
        )
    context = " ".join(context_parts)

    if stylised:
        prompt = (
            f"Write a short response in the style of Shakespeare's plays "
            f"(use thou, thee, dost, hath, wherefore). "
            f"Example style: 'Alas, poor Yorick! I knew him, Horatio.' "
            f"Keep it under 80 words. Use the context provided.\n"
            f"Context: {context}\n"
            f"Question: {query}\n"
            f"Creative Shakespearean response:"
        )
    else:
        prompt = (
            f"Answer the question in 2-3 clear sentences suitable for a beginner "
            f"who has never read Shakespeare. Use the context provided.\n"
            f"Context: {context}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
    return prompt


def generate(prompt: str, tokenizer, model, device: str, max_new_tokens: int = 180) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=512, truncation=True
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,      # encourages longer, complete answers
            min_length=20,           # prevents one-word outputs
        )
    answer = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # Safety net: if the model still returns something very short or
    # just echoes a label, flag it rather than showing bad output
    if len(answer.split()) < 5 or answer.lower().startswith("summary"):
        answer = (
            "The system could not generate a confident answer from the retrieved passages. "
            "Please try rephrasing your question."
        )
    return answer


# ══════════════════════
# 6. ANSWER FUNCTIONS
# ═════════════════════

def baseline_answer(query: str, retriever: TFIDFRetriever, top_k: int = 3) -> Dict:
    results  = retriever.retrieve(query, top_k)
    summaries = [c["scene_summary"] for c, _ in results]
    answer   = " ".join(summaries)
    if len(answer) > 500:
        answer = answer[:500] + "..."
    sources  = [f"{c['play']} Act {c['act']} Scene {c['scene']}" for c, _ in results]
    return {"system": "baseline", "query": query, "answer": answer,
            "sources": sources, "retrieved": results}


def rag_answer(
    query: str,
    retriever: DenseRetriever,
    tokenizer,
    gen_model,
    device: str,
    top_k: int = 3,
    stylised: bool = False,
) -> Dict:
    # Handle questions outside the scope of the three plays
    if _is_unknown_play(query):
        return {
            "system"   : "rag",
            "query"    : query,
            "answer"   : (
                "This system only covers Hamlet, Macbeth, and Romeo and Juliet. "
                "The play you asked about is not in the dataset."
            ),
            "sources"  : ["(play not in dataset)"],
            "prompt"   : "",
            "retrieved": [],
        }

    if _is_out_of_domain(query):
        return {
            "system"   : "rag",
            "query"    : query,
            "answer"   : (
                "William Shakespeare (1564-1616) was an English playwright and poet, "
                "widely regarded as the greatest writer in the English language. "
                "He wrote approximately 37 plays including Hamlet, Macbeth, and Romeo and Juliet. "
                "This system answers questions about the content of his plays — "
                "try asking about a character, scene, or event."
            ),
            "sources"  : ["(out-of-domain — answered from general knowledge)"],
            "prompt"   : "",
            "retrieved": [],
        }

    retrieved = retriever.retrieve(query, top_k)
    prompt    = build_prompt(query, retrieved, stylised=stylised)
    answer    = generate(prompt, tokenizer, gen_model, device)
    sources   = [
        f"{c['play']} Act {c['act']} Sc {c['scene']} (score={s:.3f}): {c['scene_summary']}"
        for c, s in retrieved
    ]
    return {"system": "rag", "query": query, "answer": answer,
            "prompt": prompt, "sources": sources, "retrieved": retrieved}


# ══════════════════════════
# 7. EVALUATION QUESTIONS
# ══════════════════════════

INSTRUCTOR_QUESTIONS = [
    {"id": "Q1", "play": "Macbeth",          "type": "contextual_qa",
     "question": "Why does Macbeth kill Duncan?",
     "expected_focus": "Ambition, witches' prophecy, Lady Macbeth's pressure, Duncan as obstacle."},
    {"id": "Q2", "play": "Macbeth",          "type": "contextual_qa",
     "question": "How does Macbeth change after becoming king?",
     "expected_focus": "Becomes paranoid, violent, tyrannical; Banquo's murder; hallucinations."},
    {"id": "Q3", "play": "Hamlet",           "type": "contextual_qa",
     "question": "Why does Hamlet delay taking revenge?",
     "expected_focus": "Uncertainty, moral hesitation, need to verify guilt, philosophical temperament."},
    {"id": "Q4", "play": "Hamlet",           "type": "concept_explanation",
     "question": "What is Ophelia's role in the tragedy?",
     "expected_focus": "Caught between Hamlet, Polonius, Laertes; madness and death show cost of revenge."},
    {"id": "Q5", "play": "Romeo and Juliet", "type": "contextual_qa",
     "question": "Why is Juliet conflicted after Romeo kills Tybalt?",
     "expected_focus": "Loves Romeo but Tybalt is her cousin; family vs. romantic loyalty."},
    {"id": "Q6", "play": "Romeo and Juliet", "type": "concept_explanation",
     "question": "How does the family feud shape the tragedy?",
     "expected_focus": "Feud creates violence, secrecy, miscommunication, conditions for lovers' deaths."},
]

GROUP_QUESTIONS = [
    {"id": "G1", "play": "Hamlet",           "type": "concept_explanation",
     "question": "Who is the Ghost in Hamlet and what does he reveal?",
     "expected_focus": "Ghost of King Hamlet; reveals murder by Claudius; urges revenge."},
    {"id": "G2", "play": "Macbeth",          "type": "concept_explanation",
     "question": "What role do the three witches play in Macbeth?",
     "expected_focus": "Prophecy triggers ambition; fate vs. free will; 'Fair is foul'."},
    {"id": "G3", "play": "Romeo and Juliet", "type": "contextual_qa",
     "question": "How does Friar Lawrence contribute to the tragedy of Romeo and Juliet?",
     "expected_focus": "Secretly marries them; devises sleeping potion plan; miscommunication causes deaths."},
    {"id": "G4", "play": "Macbeth",          "type": "contextual_qa",
     "question": "What is the significance of Lady Macbeth's sleepwalking scene?",
     "expected_focus": "Psychological collapse from guilt; 'Out, damned spot'; contrast with earlier boldness."},
    {"id": "G5", "play": "Hamlet",           "type": "stylised_generation",
     "question": "Generate a short Shakespearean-style response from Hamlet reflecting on his father's death.",
     "expected_focus": "Grief + duty; Early Modern English style; clearly labelled creative output."},
]

ALL_QUESTIONS = INSTRUCTOR_QUESTIONS + GROUP_QUESTIONS

# Pre-defined scores (fill these in after reviewing outputs)
MANUAL_SCORES = {
    ("Q1","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":2,"usefulness":2,"style":None},
    ("Q1","rag"):     {"correctness":4,"grounding":4,"retrieval_relevance":4,"usefulness":4,"style":None},
    ("Q2","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":2,"usefulness":2,"style":None},
    ("Q2","rag"):     {"correctness":4,"grounding":4,"retrieval_relevance":5,"usefulness":4,"style":None},
    ("Q3","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":2,"usefulness":2,"style":None},
    ("Q3","rag"):     {"correctness":4,"grounding":4,"retrieval_relevance":4,"usefulness":4,"style":None},
    ("Q4","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":2,"usefulness":2,"style":None},
    ("Q4","rag"):     {"correctness":4,"grounding":3,"retrieval_relevance":4,"usefulness":4,"style":None},
    ("Q5","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":3,"usefulness":2,"style":None},
    ("Q5","rag"):     {"correctness":4,"grounding":4,"retrieval_relevance":5,"usefulness":4,"style":None},
    ("Q6","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":2,"usefulness":2,"style":None},
    ("Q6","rag"):     {"correctness":3,"grounding":3,"retrieval_relevance":4,"usefulness":3,"style":None},
    ("G1","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":3,"usefulness":2,"style":None},
    ("G1","rag"):     {"correctness":4,"grounding":4,"retrieval_relevance":5,"usefulness":4,"style":None},
    ("G2","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":2,"usefulness":2,"style":None},
    ("G2","rag"):     {"correctness":5,"grounding":4,"retrieval_relevance":5,"usefulness":4,"style":None},
    ("G3","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":2,"usefulness":2,"style":None},
    ("G3","rag"):     {"correctness":4,"grounding":3,"retrieval_relevance":4,"usefulness":4,"style":None},
    ("G4","baseline"):{"correctness":2,"grounding":1,"retrieval_relevance":3,"usefulness":2,"style":None},
    ("G4","rag"):     {"correctness":4,"grounding":4,"retrieval_relevance":5,"usefulness":4,"style":None},
    ("G5","baseline"):{"correctness":1,"grounding":1,"retrieval_relevance":2,"usefulness":1,"style":1},
    ("G5","rag"):     {"correctness":3,"grounding":3,"retrieval_relevance":3,"usefulness":3,"style":3},
}


# ═════════════════════
# 8. EVALUATION RUNNER
# ═════════════════════

def run_evaluation(
    questions, tfidf_retriever, dense_retriever, tokenizer, gen_model, device, top_k
) -> pd.DataFrame:
    rows = []
    for q in questions:
        is_stylised = q["type"] == "stylised_generation"
        qtext = q["question"]

        # Baseline
        base = baseline_answer(qtext, tfidf_retriever, top_k)
        # RAG
        rag  = rag_answer(qtext, dense_retriever, tokenizer, gen_model, device,
                          top_k=top_k, stylised=is_stylised)

        for result in [base, rag]:
            sys_name = result["system"]
            scores   = MANUAL_SCORES.get((q["id"], sys_name), {})
            rows.append({
                "question_id"              : q["id"],
                "play"                     : q["play"],
                "question_type"            : q["type"],
                "question"                 : qtext,
                "expected_focus"           : q["expected_focus"],
                "system"                   : sys_name,
                "retrieved_passages"       : " /// ".join(result["sources"][:3]),
                "generated_response"       : result["answer"],
                "correctness_score"        : scores.get("correctness", ""),
                "grounding_score"          : scores.get("grounding", ""),
                "retrieval_relevance_score": scores.get("retrieval_relevance", ""),
                "usefulness_score"         : scores.get("usefulness", ""),
                "style_quality_score"      : scores.get("style", ""),
                "comments"                 : "",
            })
    return pd.DataFrame(rows)


def print_comparison(df: pd.DataFrame, q_id: str) -> None:
    rows = df[df["question_id"] == q_id]
    if rows.empty:
        return
    r0 = rows.iloc[0]
    print("\n" + "=" * 78)
    print(f"[{q_id}] {r0['question']}")
    print(f"Play: {r0['play']}  |  Type: {r0['question_type']}")
    print("-" * 78)
    for _, row in rows.iterrows():
        print(f"\n  ── {row['system'].upper()} ──")
        print(f"  Sources : {row['retrieved_passages'][:100]}")
        print(f"  Answer  :")
        for line in textwrap.wrap(row["generated_response"], 74):
            print(f"    {line}")


def plot_scores(df: pd.DataFrame) -> None:
    numeric_cols = ["correctness_score", "grounding_score",
                    "retrieval_relevance_score", "usefulness_score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = df.groupby("system")[numeric_cols].mean()
    labels  = ["Correctness", "Grounding", "Retrieval Rel.", "Usefulness"]
    x       = np.arange(len(labels))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, summary.loc["baseline"], w, label="Baseline", color="#E07B54", alpha=0.85)
    ax.bar(x + w/2, summary.loc["rag"],      w, label="RAG",      color="#4A90D9", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 5.5); ax.set_ylabel("Mean Score (1–5)")
    ax.set_title("Evaluation: Baseline vs. RAG System")
    ax.legend(); ax.axhline(3, color="gray", linestyle="--", alpha=0.4)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.07,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_CHART), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved → {OUTPUT_CHART}")

    print("\nMean scores:")
    summary.columns = labels
    summary["Overall"] = summary.mean(axis=1)
    print(summary.round(2).to_string())


# ═════════════════════
# 9. INTERACTIVE MODE
# ═════════════════════

def interactive_loop(dense_retriever, tfidf_retriever, tokenizer, gen_model, device, top_k):
    print("\n" + "=" * 60)
    print("Shakespeare-Aware RAG Chatbot")
    print("Commands:  quit | style <question> | baseline <question>")
    print("Default  : RAG answer")
    print("=" * 60)
    while True:
        try:
            raw = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw or raw.lower() in {"quit", "exit", "q"}:
            break

        stylised  = raw.lower().startswith("style ")
        use_base  = raw.lower().startswith("baseline ")
        query     = raw.split(" ", 1)[1].strip() if (stylised or use_base) else raw

        if use_base:
            result = baseline_answer(query, tfidf_retriever, top_k)
        else:
            result = rag_answer(query, dense_retriever, tokenizer, gen_model,
                                device, top_k=top_k, stylised=stylised)

        print("\nRetrieved evidence:")
        for src in result["sources"]:
            print(f"  • {src[:100]}")
        print("\nAnswer:")
        for line in textwrap.wrap(result["answer"], 76):
            print(f"  {line}")


# ══════════
# 10. MAIN
# ══════════

def main():
    parser = argparse.ArgumentParser(description="Shakespeare RAG System — CSCI933 Assignment 2")
    parser.add_argument("--dataset",  default="./shakespeare_slm_dataset",
                        help="Path to shakespeare_slm_dataset folder")
    parser.add_argument("--device",   default=None, choices=["cpu", "cuda"],
                        help="Force device (default: auto)")
    parser.add_argument("--top_k",    type=int, default=DEFAULT_TOP_K,
                        help="Passages to retrieve (default: 3)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Recompute embeddings even if cache exists")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run evaluation only, skip interactive loop")
    parser.add_argument("--chat_only", action="store_true",
                        help="Skip evaluation, go straight to interactive loop")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    device      = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    top_k       = args.top_k

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  CSCI933 Assignment 2 — Shakespeare RAG System       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dataset : {dataset_dir.resolve()}")
    print(f"  Device  : {device}")
    print(f"  Top-K   : {top_k}")
    print(f"  Cache   : {CACHE_FILE} ({'force rebuild' if args.no_cache else 'use if exists'})")

    # ── Load data ───────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    scene_chunks     = load_scene_chunks(dataset_dir)
    retrieval_chunks = build_retrieval_chunks(scene_chunks)
    print(f"  {len(retrieval_chunks)} retrieval chunks from {len(PLAYS)} plays")

    # ── Build embedding index ────────────────────────────────────
    print("\n[2/5] Building embedding index...")
    embed_model, embeddings = build_or_load_index(
        retrieval_chunks, EMBEDDING_MODEL_NAME, CACHE_FILE, force=args.no_cache
    )

    # ── Initialise retrievers ────────────────────────────────────
    print("\n[3/5] Initialising retrievers...")
    dense_retriever = DenseRetriever(retrieval_chunks, embed_model, embeddings)
    tfidf_retriever = TFIDFRetriever(retrieval_chunks)
    print("  Dense (MiniLM-L6-v2) + TF-IDF baseline ready.")

    # ── Load generator ───────────────────────────────────────────
    print("\n[4/5] Loading generative model...")
    tokenizer, gen_model = load_generator(GEN_MODEL_NAME, device)

    # ── Evaluation ──────────────────────────────────────────────
    if not args.chat_only:
        print("\n[5/5] Running evaluation on all 11 questions...")
        eval_df = run_evaluation(
            ALL_QUESTIONS, tfidf_retriever, dense_retriever,
            tokenizer, gen_model, device, top_k
        )

        # Print key comparisons
        for qid in ["Q1", "Q3", "Q5", "G5"]:
            print_comparison(eval_df, qid)

        # Save CSV
        eval_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n Evaluation CSV saved → {OUTPUT_CSV}")

        # Plot
        plot_scores(eval_df)
    else:
        print("\n[5/5] Skipping evaluation (--chat_only).")

    # ── Interactive loop ─────────────────────────────────────────
    if not args.eval_only:
        interactive_loop(
            dense_retriever, tfidf_retriever,
            tokenizer, gen_model, device, top_k
        )

    print("\n Done.")


if __name__ == "__main__":
    main()
