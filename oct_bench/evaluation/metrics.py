"""
oct_bench.evaluation.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Advanced evaluation metrics:
  - BLEU-1/2/4  (via nltk)
  - ROUGE-L      (via rouge-score)
  - Hallucination Rate  (keyword-overlap heuristic)
  - Expected Calibration Error (ECE)
"""

from __future__ import annotations
import math
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i: i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> Dict[str, float]:
    """
    Sentence-level BLEU-1 … BLEU-{max_n}.
    Returns a dict like {"BLEU-1": 0.72, "BLEU-2": 0.55, ...}
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    scores: Dict[str, float] = {}
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hyp_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)

        clipped = sum(
            min(cnt, ref_ngrams.get(gram, 0))
            for gram, cnt in hyp_ngrams.items()
        )
        total = max(sum(hyp_ngrams.values()), 1)
        precision = clipped / total

        # Brevity penalty
        bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)) \
            if len(hyp_tokens) < len(ref_tokens) else 1.0

        scores[f"BLEU-{n}"] = round(bp * precision * 100, 2)
    return scores


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(hypothesis: str, reference: str, beta: float = 1.2) -> float:
    """
    Sentence-level ROUGE-L F-score (0–100).
    beta > 1 weights recall higher (standard for summarization).
    """
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    f = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    return round(f * 100, 2)


# ---------------------------------------------------------------------------
# Hallucination Rate
# ---------------------------------------------------------------------------

# Clinical terms that should appear in OCT explanations when present in reference
_CLINICAL_KEYWORDS = {
    "retina", "oct", "fluid", "lesion", "layer", "irf", "srf", "ped",
    "macula", "fovea", "choroid", "drusen", "edema", "atrophy",
    "reflectivity", "hyperreflective", "hyporeflective", "epiretinal",
    "membrane", "detachment", "neovascular", "amd", "dme", "normal",
    "pathology", "disease", "diagnosis", "treatment", "staging",
}


def hallucination_rate(
    predictions: List[str],
    references: List[str],
    threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Heuristic hallucination detector based on clinical keyword overlap.

    A prediction is flagged as a hallucination if:
      - The reference contains ≥1 clinical keyword AND
      - The overlap ratio between reference keywords and prediction keywords < threshold.

    Returns:
        {
            "hallucination_rate": float (0–100%),
            "flagged_count": int,
            "total_count": int,
        }
    """
    flagged = 0
    for pred, ref in zip(predictions, references):
        ref_words = set(ref.lower().split())
        pred_words = set(pred.lower().split())

        ref_kws = ref_words & _CLINICAL_KEYWORDS
        if not ref_kws:
            continue  # reference has no clinical terms → skip

        pred_kws = pred_words & _CLINICAL_KEYWORDS
        overlap = len(ref_kws & pred_kws) / len(ref_kws)
        if overlap < threshold:
            flagged += 1

    total = len(predictions)
    rate = (flagged / total * 100) if total > 0 else 0.0
    return {
        "hallucination_rate": round(rate, 2),
        "flagged_count": flagged,
        "total_count": total,
    }


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: List[float],
    correctness: List[bool],
    n_bins: int = 10,
) -> float:
    """
    Computes ECE (0–100). Lower is better (model confidence ≈ actual accuracy).

    Args:
        confidences: Model confidence per prediction (0–1).
        correctness: Whether prediction was correct.
        n_bins: Number of calibration bins.
    """
    if not confidences:
        return 0.0
    bin_size = 1.0 / n_bins
    ece = 0.0
    n = len(confidences)
    for b in range(n_bins):
        lo = b * bin_size
        hi = lo + bin_size
        indices = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not indices:
            continue
        avg_conf = sum(confidences[i] for i in indices) / len(indices)
        avg_acc = sum(1 for i in indices if correctness[i]) / len(indices)
        ece += (len(indices) / n) * abs(avg_conf - avg_acc)
    return round(ece * 100, 2)


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    correctness: List[bool],
    confidences: List[float] | None = None,
) -> Dict[str, float]:
    """
    Convenience function: run all metrics over parallel lists.

    Args:
        predictions : model output texts (or MCQ letter predictions)
        references  : ground-truth texts / correct answers
        correctness : True if prediction matched ground truth
        confidences : optional confidence scores [0,1]

    Returns flat dict of all metric values.
    """
    results: Dict[str, float] = {}

    # BLEU & ROUGE over all pairs → macro average
    bleu_acc: Dict[str, float] = {"BLEU-1": 0, "BLEU-2": 0, "BLEU-4": 0}
    rouge_acc = 0.0
    for p, r in zip(predictions, references):
        b = bleu_score(p, r, max_n=4)
        bleu_acc["BLEU-1"] += b.get("BLEU-1", 0)
        bleu_acc["BLEU-2"] += b.get("BLEU-2", 0)
        bleu_acc["BLEU-4"] += b.get("BLEU-4", 0)
        rouge_acc += rouge_l(p, r)

    n = max(len(predictions), 1)
    results["BLEU-1"] = round(bleu_acc["BLEU-1"] / n, 2)
    results["BLEU-2"] = round(bleu_acc["BLEU-2"] / n, 2)
    results["BLEU-4"] = round(bleu_acc["BLEU-4"] / n, 2)
    results["ROUGE-L"] = round(rouge_acc / n, 2)

    # Hallucination rate
    hall = hallucination_rate(predictions, references)
    results["Hallucination Rate (%)"] = hall["hallucination_rate"]
    results["Flagged Hallucinations"] = float(hall["flagged_count"])

    # Accuracy
    results["Accuracy (%)"] = round(sum(correctness) / max(len(correctness), 1) * 100, 2)

    # ECE (only if confidences provided)
    if confidences:
        results["ECE (%)"] = expected_calibration_error(confidences, correctness)

    return results
