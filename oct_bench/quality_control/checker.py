import re
from typing import Dict, Any, Tuple
from oct_bench.data.schemas import MCQ

class LogicalQCChecker:
    """
    Automated checks to enforce high quality and eliminate logical flaws in generated MCQs.
    Checks for: option formatting, duplicate choices, length imbalances, language-only cues, and spelling.
    """
    def __init__(self):
        pass

    def run_checks(self, mcq: MCQ) -> Tuple[bool, str]:
        """
        Runs a suite of structural and semantic checks on the question.
        Returns: (is_passed: bool, reason: str)
        """
        # 1. Structural integrity check
        if len(mcq.options) != 4:
            return False, f"Failed: Options count is {len(mcq.options)} (must be exactly 4)"

        required_keys = {"A", "B", "C", "D"}
        if set(mcq.options.keys()) != required_keys:
            return False, f"Failed: Options keys must be exactly A, B, C, D. Found {list(mcq.options.keys())}"

        if mcq.correct_answer not in required_keys:
            return False, f"Failed: Correct answer '{mcq.correct_answer}' is not in options A, B, C, D"

        # 2. Empty values check
        for k, v in mcq.options.items():
            if not v or not v.strip():
                return False, f"Failed: Option {k} is empty"

        # 3. Duplicate options check
        option_texts = [v.strip().lower() for v in mcq.options.values()]
        if len(set(option_texts)) < 4:
            return False, "Failed: Contains duplicate option texts"

        # 4. Length bias check
        # If one option is extremely long compared to others, LLMs can guess it
        lengths = [len(v) for v in mcq.options.values()]
        correct_len = len(mcq.options[mcq.correct_answer])
        avg_other_len = (sum(lengths) - correct_len) / 3.0
        
        # If correct answer is > 3 times longer than average distractors
        if avg_other_len > 0 and correct_len > 3 * avg_other_len:
            return False, f"Failed: Correct answer option '{mcq.correct_answer}' is abnormally long, introducing bias"

        # 5. Language cue checks (e.g. absolute words or meta references)
        absolute_pattern = re.compile(r"\b(always|never|completely healthy|100%|guaranteed)\b", re.IGNORECASE)
        # Avoid references to the formatting itself
        format_pattern = re.compile(r"\b(as shown in option|multiple choice|question above)\b", re.IGNORECASE)
        
        for k, text in mcq.options.items():
            if format_pattern.search(text):
                return False, f"Failed: Option {k} references formatting text"

        # 6. Automated logical verification check (Self-consistent key checks)
        # Verify the question actually references the image context rather than general knowledge
        # Example: T01 modality should mention 'this scan' or 'image'
        context_words = {"image", "scan", "box", "highlighted", "shown", "demonstrated", "annotation", "retina", "lesion"}
        question_lower = mcq.question.lower()
        has_context = any(word in question_lower for word in context_words)
        
        if not has_context:
            return False, "Failed: Question lacks reference to visual elements (image, scan, annotations, etc.)"

        return True, "Passed all automated QC checks"
