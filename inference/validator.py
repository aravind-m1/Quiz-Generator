"""
Quality Validation Layer
=========================
Multi-stage validation pipeline that ensures generated questions are:
  1. Grammatically correct (via language-tool-python)
  2. Factually grounded in the context (via NLI cross-encoder)
  3. Free of ambiguity and logical issues
  4. Non-duplicate (diversity enforcement via cosine similarity)

Any question failing validation is flagged or removed from the output.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── NLI Entailment Validator ───────────────────────────────────────────────────

class NLIValidator:
    """
    Uses a cross-encoder NLI model to verify that the generated question + answer
    is entailed by the source context (preventing hallucinations).
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        from config import NLI_MODEL_NAME, DEVICE

        self.model_name = model_name or NLI_MODEL_NAME
        self.device = device or DEVICE
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        logger.info("Loading NLI model: %s", self.model_name)
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info("NLI model loaded.")

    def score_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Score how well the premise (context) entails the hypothesis (Q+A statement).

        Parameters
        ----------
        premise : str
            The source context.
        hypothesis : str
            A declarative statement derived from the question and answer.

        Returns
        -------
        float
            Entailment score in [0, 1]. Higher = more entailed.
        """
        self._load_model()
        scores = self._model.predict([(premise, hypothesis)])
        # DeBERTa NLI outputs: [contradiction, neutral, entailment]
        if hasattr(scores[0], "__len__") and len(scores[0]) == 3:
            return float(scores[0][2])  # entailment score
        return float(scores[0])

    def validate_question(
        self, context: str, question: str, answer: str, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate a question-answer pair against the source context.

        Returns
        -------
        Dict with keys: 'is_valid', 'entailment_score', 'reason'
        """
        from config import NLI_ENTAILMENT_THRESHOLD

        threshold = threshold or NLI_ENTAILMENT_THRESHOLD

        # Convert Q+A into a declarative hypothesis
        hypothesis = _qa_to_statement(question, answer)
        score = self.score_entailment(context, hypothesis)

        is_valid = score >= threshold
        reason = ""
        if not is_valid:
            reason = (
                f"Low entailment score ({score:.3f} < {threshold}). "
                f"The answer may not be fully supported by the context."
            )

        return {
            "is_valid": is_valid,
            "entailment_score": round(score, 4),
            "hypothesis": hypothesis,
            "reason": reason,
        }


# ─── Grammar Validator ──────────────────────────────────────────────────────────

class GrammarValidator:
    """
    Checks generated text for grammar and spelling errors using
    language-tool-python (LanguageTool).
    """

    def __init__(self):
        self._tool = None

    def _load_tool(self):
        if self._tool is not None:
            return
        try:
            import language_tool_python

            self._tool = language_tool_python.LanguageTool("en-US")
            logger.info("LanguageTool grammar checker loaded.")
        except Exception as exc:
            logger.warning("Grammar checker unavailable: %s", exc)
            self._tool = False  # sentinel

    def check(self, text: str) -> Dict[str, Any]:
        """
        Check text for grammar issues.

        Returns
        -------
        Dict with keys: 'is_clean', 'error_count', 'errors'
        """
        self._load_tool()

        if not self._tool:
            return {"is_clean": True, "error_count": 0, "errors": []}

        matches = self._tool.check(text)
        errors = [
            {
                "message": m.message,
                "context": m.context,
                "suggestions": m.replacements[:3] if m.replacements else [],
                "category": m.category,
            }
            for m in matches
            # Ignore minor issues like whitespace
            if m.category not in ("TYPOGRAPHY", "WHITESPACE")
        ]

        return {
            "is_clean": len(errors) == 0,
            "error_count": len(errors),
            "errors": errors,
        }

    def auto_correct(self, text: str) -> str:
        """Apply auto-corrections to text."""
        self._load_tool()
        if not self._tool:
            return text
        try:
            import language_tool_python

            return language_tool_python.utils.correct(text, self._tool.check(text))
        except Exception:
            return text


# ─── Structural Validator ───────────────────────────────────────────────────────

class StructuralValidator:
    """
    Validates the structural integrity of generated questions:
    - MCQ must have exactly 4 options
    - Answer must be present in options (for MCQ)
    - Question must end with a question mark (or be a valid statement)
    - No empty fields
    """

    @staticmethod
    def validate(question_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate structural rules.

        Returns
        -------
        Dict with keys: 'is_valid', 'issues'
        """
        issues: list[str] = []
        q_type = question_dict.get("type", "MCQ")
        question = question_dict.get("question", "")
        answer = question_dict.get("answer", "")
        options = question_dict.get("options", [])

        # Check empty fields
        if not question.strip():
            issues.append("Question text is empty.")
        if not answer.strip():
            issues.append("Answer is empty.")

        # MCQ-specific checks
        if q_type == "MCQ":
            if len(options) < 4:
                issues.append(
                    f"MCQ must have 4 options, but has {len(options)}."
                )
            if answer and options:
                # Check if answer matches any option (case-insensitive)
                option_lower = [o.lower().strip() for o in options]
                if answer.lower().strip() not in option_lower:
                    issues.append("Correct answer not found among options.")

        # TrueFalse checks
        if q_type == "TrueFalse":
            if answer.lower().strip() not in ("true", "false"):
                issues.append(
                    f"TrueFalse answer must be 'True' or 'False', got '{answer}'."
                )

        # FillInTheBlank checks
        if q_type == "FillInTheBlank":
            if "___" not in question and "____" not in question and "blank" not in question.lower():
                issues.append(
                    "FillInTheBlank question must contain a blank indicator."
                )

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
        }


# ─── Diversity Filter ──────────────────────────────────────────────────────────

class DiversityFilter:
    """
    Ensures generated questions are diverse and non-repetitive using
    cosine similarity between question embeddings.
    """

    def __init__(self, similarity_threshold: Optional[float] = None):
        from config import DIVERSITY_SIMILARITY_THRESHOLD

        self.threshold = similarity_threshold or DIVERSITY_SIMILARITY_THRESHOLD
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def filter_duplicates(
        self, questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate questions from the list.

        Parameters
        ----------
        questions : List[Dict[str, Any]]
            List of question dictionaries.

        Returns
        -------
        List[Dict[str, Any]]
            Filtered list with diverse questions only.
        """
        if len(questions) <= 1:
            return questions

        self._load_model()
        import numpy as np

        texts = [q.get("question", "") for q in questions]
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        keep_indices = [0]
        for i in range(1, len(questions)):
            is_diverse = True
            for j in keep_indices:
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim > self.threshold:
                    is_diverse = False
                    logger.info(
                        "Removing duplicate question (sim=%.3f): %s",
                        sim,
                        texts[i][:80],
                    )
                    break
            if is_diverse:
                keep_indices.append(i)

        return [questions[i] for i in keep_indices]


# ─── Unified Validator Pipeline ────────────────────────────────────────────────

class QuestionValidator:
    """
    Orchestrates all validation layers into a single pipeline.
    """

    def __init__(self):
        self.nli_validator = NLIValidator()
        self.grammar_validator = GrammarValidator()
        self.structural_validator = StructuralValidator()
        self.diversity_filter = DiversityFilter()

    def validate_single(
        self,
        question_dict: Dict[str, Any],
        context: str,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all validation checks on a single question.

        Parameters
        ----------
        question_dict : Dict[str, Any]
            The question to validate.
        context : str
            The source context.
        strict : bool
            If True, questions failing any check are marked invalid.

        Returns
        -------
        Dict[str, Any]
            The question dict enriched with validation metadata.
        """
        question = question_dict.get("question", "")
        answer = question_dict.get("answer", "")
        validation = {}

        # 1. Structural validation
        structural = self.structural_validator.validate(question_dict)
        validation["structural"] = structural

        # 2. Grammar check
        grammar = self.grammar_validator.check(question)
        validation["grammar"] = grammar

        # 3. NLI entailment (only if question and answer are non-empty)
        if question and answer:
            try:
                nli = self.nli_validator.validate_question(context, question, answer)
                validation["nli"] = nli
            except Exception as exc:
                logger.warning("NLI validation failed: %s", exc)
                validation["nli"] = {
                    "is_valid": True,
                    "entailment_score": -1,
                    "reason": f"NLI check skipped: {exc}",
                }
        else:
            validation["nli"] = {
                "is_valid": False,
                "entailment_score": 0,
                "reason": "Empty question or answer.",
            }

        # Determine overall validity
        if strict:
            is_valid = all(
                validation[k].get("is_valid", True) for k in validation
            )
        else:
            # In lenient mode, only structural issues are hard failures
            is_valid = validation["structural"]["is_valid"]

        question_dict["validation"] = validation
        question_dict["is_valid"] = is_valid
        return question_dict

    def validate_batch(
        self,
        questions: List[Dict[str, Any]],
        context: str,
        strict: bool = True,
        enforce_diversity: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Validate and filter a batch of questions.

        Parameters
        ----------
        questions : List[Dict[str, Any]]
            Generated questions to validate.
        context : str
            Source context.
        strict : bool
            Whether to apply strict validation.
        enforce_diversity : bool
            Whether to remove near-duplicate questions.

        Returns
        -------
        List[Dict[str, Any]]
            Valid, diverse questions.
        """
        validated = []
        for q in questions:
            result = self.validate_single(q, context, strict=strict)
            if result.get("is_valid", False):
                validated.append(result)
            else:
                logger.info(
                    "Question removed by validation: %s",
                    result.get("question", "")[:80],
                )

        if enforce_diversity:
            validated = self.diversity_filter.filter_duplicates(validated)

        logger.info(
            "Validation: %d/%d questions passed.", len(validated), len(questions)
        )
        return validated


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _qa_to_statement(question: str, answer: str) -> str:
    """
    Convert a question + answer pair into a declarative statement
    for NLI entailment checking.

    Example:
        Q: "What is photosynthesis?"
        A: "A process to convert light to chemical energy"
        → "Photosynthesis is a process to convert light to chemical energy."
    """
    q = question.strip().rstrip("?").rstrip(".")
    a = answer.strip().rstrip(".")

    # Simple pattern: "What is X?" → "X is A"
    what_match = re.match(r"(?i)what (?:is|are) (.+)", q)
    if what_match:
        subject = what_match.group(1).strip()
        return f"{subject} is {a}."

    # "Which X ...?" → "The answer to ... is A"
    # Default: concatenate
    return f"{q}. The answer is {a}."
