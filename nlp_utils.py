"""NLP utilities: spaCy NER keyword extraction, ROUGE scoring, quality metrics.

All heavy dependencies are loaded lazily and fall back gracefully so the app
starts even if spaCy models or the evaluate library are unavailable.
"""

from typing import List, Optional
import re

# ── Lazy singletons ──────────────────────────────────────────────────────────

_spacy_nlp = None
_rouge_metric = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess, sys
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    capture_output=True,
                )
                _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            pass  # fall back to NLTK
    return _spacy_nlp


def _get_rouge():
    global _rouge_metric
    if _rouge_metric is None:
        try:
            import evaluate
            _rouge_metric = evaluate.load("rouge")
        except Exception:
            pass
    return _rouge_metric


# ── Keyword extraction ───────────────────────────────────────────────────────

def extract_keywords(text: str) -> List[str]:
    """Extract keywords via spaCy NER + noun chunks, falling back to NLTK."""
    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 2]
        chunks = [
            chunk.root.lemma_.lower()
            for chunk in doc.noun_chunks
            if len(chunk.text) > 3
        ]
        # deduplicate while preserving order
        seen, keywords = set(), []
        for w in entities + chunks:
            if w not in seen:
                seen.add(w)
                keywords.append(w)
        return keywords[:8] if keywords else _nltk_keywords(text)
    return _nltk_keywords(text)


def _nltk_keywords(text: str) -> List[str]:
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        nltk.download("stopwords", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 3][:5]
    except Exception:
        return [w for w in text.lower().split() if len(w) > 3][:5]


# ── ROUGE scoring ────────────────────────────────────────────────────────────

def compute_rouge(generated: str, reference: Optional[str] = None) -> dict:
    """
    Compute ROUGE-1/2/L scores.

    If no reference is provided the intro paragraph is used as reference
    and the body paragraphs as the hypothesis — a useful self-consistency
    signal for long-form content.
    """
    rouge = _get_rouge()
    if not rouge:
        return {}
    try:
        if reference is None:
            paragraphs = [p.strip() for p in generated.split("\n\n") if p.strip()]
            if len(paragraphs) >= 2:
                reference = paragraphs[0]
                hypothesis = "\n\n".join(paragraphs[1:])
            else:
                mid = len(generated) // 2
                reference, hypothesis = generated[:mid], generated[mid:]
        else:
            hypothesis = generated

        scores = rouge.compute(predictions=[hypothesis], references=[reference])
        return {
            "rouge1": round(scores["rouge1"], 3),
            "rouge2": round(scores["rouge2"], 3),
            "rougeL": round(scores["rougeL"], 3),
        }
    except Exception:
        return {}


# ── Readability ───────────────────────────────────────────────────────────────

def compute_readability(content: str) -> dict:
    """
    Returns Flesch Reading Ease, Flesch-Kincaid grade level, and a
    human-readable grade label.  Falls back gracefully if textstat is absent.
    """
    try:
        import textstat
        ease  = round(textstat.flesch_reading_ease(content), 1)
        grade = round(textstat.flesch_kincaid_grade(content), 1)
        fog   = round(textstat.gunning_fog(content), 1)

        # Map ease score → human label
        if ease >= 90:
            label = "Very Easy (5th grade)"
        elif ease >= 70:
            label = "Easy (6th–7th grade)"
        elif ease >= 60:
            label = "Standard (8th–9th grade)"
        elif ease >= 50:
            label = "Fairly Difficult (10th–12th grade)"
        elif ease >= 30:
            label = "Difficult (College level)"
        else:
            label = "Very Difficult (Professional)"

        return {
            "flesch_ease": ease,
            "flesch_kincaid_grade": grade,
            "gunning_fog": fog,
            "reading_level": label,
        }
    except Exception:
        return {}


# ── Quality metrics ──────────────────────────────────────────────────────────

def compute_quality_metrics(content: str) -> dict:
    words = content.split()
    paragraphs = [p for p in content.split("\n\n") if p.strip()]
    sentences = len([s for s in re.split(r"[.!?]", content) if s.strip()])
    word_count = len(words)

    length_score = min(30, word_count // 20)
    structure_score = min(20, len(paragraphs) * 5)
    sentence_score = min(10, sentences // 5)
    quality_score = min(100, 40 + length_score + structure_score + sentence_score)

    return {
        "word_count": word_count,
        "paragraph_count": len(paragraphs),
        "sentence_count": sentences,
        "reading_time_minutes": max(1, round(word_count / 200)),
        "quality_score": quality_score,
        "readability": compute_readability(content),
        "rouge": compute_rouge(content),
    }
