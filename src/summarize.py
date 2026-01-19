import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import nltk
from rouge_score import rouge_scorer
import tiktoken
from dotenv import load_dotenv
from google import genai

def load_env():
    print("cwd =", Path.cwd())

    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]

    for p in candidates:
        print("checking:", p, "exists:", p.exists())

    for p in candidates:
        if p.exists():
            load_dotenv(p, override=True)
            print("loaded .env from:", p)
            return

    print("WARNING: no .env found")

def ensure_nltk():
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)



def clean_text(x: str) -> str:
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def token_len(enc, s: str) -> int:
    return len(enc.encode(s))


def gemini_generate(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
) -> str:
    client = genai.Client()
    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}],
        config={"temperature": temperature},
    )
    
    text = (resp.text or "").strip()

    # debug: why it stopped
    try:
        fr = resp.candidates[0].finish_reason
        print(f"[gemini] finish_reason={fr}, out_chars={len(text)}")
    except Exception:
        pass

    return text


def draft_summary(text: str, model: str, draft_words: int = 180) -> str:
    system = "You are a helpful assistant that writes faithful summaries."
    user = (
        "Write a concise draft summary of the following document.\n"
        f"- Target length: about {draft_words} words\n"
        "- Be faithful: do not add facts not supported by the document.\n\n"
        "DOCUMENT:\n"
        f"{text}"
    )
    return gemini_generate(
        model=model, system=system, user=user, temperature=0.2
    )


def rouge_recall_score(sent: str, target: str, metric: str) -> float:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2"], use_stemmer=True
    )
    scores = scorer.score(target, sent)
    if metric == "rouge-1":
        return scores["rouge1"].recall
    if metric == "rouge-2":
        return scores["rouge2"].recall
    if metric == "rouge-1+2":
        return scores["rouge1"].recall + scores["rouge2"].recall
    raise ValueError()


def extract_top_sentences_by_rouge(
    source_text: str,
    target_text: str,
    metric: str = "rouge-1+2",
    budget_tokens: int = 1400,
) -> Tuple[str, List[int]]:
    ensure_nltk()
    enc = tiktoken.get_encoding("cl100k_base")

    sents = nltk.sent_tokenize(source_text)
    if not sents:
        return source_text, list(range(len(source_text)))

    scored = []
    for i, sent in enumerate(sents):
        sc = rouge_recall_score(sent, target_text, metric)
        scored.append((sc, i))

    scored.sort(reverse=True, key=lambda x: x[0])

    chosen = []
    used = 0
    for sc, i in scored:
        t = token_len(enc, sents[i] + "\n")
        if used + t > budget_tokens:
            continue
        chosen.append(i)
        used += t
        if used >= budget_tokens:
            break

    chosen.sort()
    extracted = " ".join(sents[i] for i in chosen)
    return extracted, chosen


def final_summary_from_extracted(extracted: str, model: str, out_words: int = 220) -> str:
    system = "You are a helpful assistant that writes faithful summaries."
    user = (
        "Write a high-quality summary using ONLY the information in the provided extracted sentences.\n"
        f"- Target length: about {out_words} words\n"
        "- Do not add any facts not supported by the extracted text.\n"
        "- Prefer clear structure (2-5 paragraphs or bullets if appropriate).\n\n"
        "EXTRACTED SENTENCES:\n"
        f"{extracted}"
    )
    return gemini_generate(
        model=model, system=system, user=user, temperature=0.2)


def main():
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_txt", required=True)
    parser.add_argument("--out_txt", required=True)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--metric", default="rouge-1+2", choices=["rouge-1", "rouge-2", "rouge-1+2"]
    )
    parser.add_argument("--extract_budget_tokens", type=int, default=1400)
    parser.add_argument("--draft_words", type=int, default=180)
    parser.add_argument("--final_words", type=int, default=220)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")

    text = clean_text(open(args.in_txt, "r", encoding="utf-8").read())

    draft = draft_summary(text, model=args.model, draft_words=args.draft_words)

    extracted, idx = extract_top_sentences_by_rouge(
        source_text=text,
        target_text=draft,
        metric=args.metric,
        budget_tokens=args.extract_budget_tokens,
    )

    final = final_summary_from_extracted(
        extracted, model=args.model, out_words=args.final_words
    )

    out = []
    # out.append("### DRAFT SUMMARY (Pass 1)\n")
    # out.append(draft.strip() + "\n")
    # out.append("\n### EXTRACTED SENTENCES (Pass 2)\n")
    # out.append(extracted.strip() + "\n")
    # out.append("\n### FINAL SUMMARY (Pass 3)\n")
    out.append(final.strip() + "\n")
    # out.append("\n### EXTRACTED SENTENCE INDICES\n")
    # out.append(str(idx) + "\n")

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote: {args.out_txt}")


if __name__ == "__main__":
    main()
