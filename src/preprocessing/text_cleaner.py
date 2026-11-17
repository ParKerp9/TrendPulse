"""Text Cleaning, NLP Normalization, and Sentiment Processing for Social Media."""

import html
import re
from typing import Any, Dict, List, Optional, Tuple


class TextCleaner:
    """Preprocesses social media text: removes artifacts, extracts linguistic signals, and computes sentiment."""

    # Common positive and negative sentiment lexicons for lightweight, high-performance scoring
    POSITIVE_WORDS = {
        "breakthrough", "record", "surge", "gain", "soar", "rally", "outperform", "milestone",
        "revolutionary", "innovative", "scalable", "efficient", "faster", "improved", "remarkable",
        "repeatability", "advance", "optimal", "incredible", "promising", "success", "superior",
        "flawless", "lucrative", "bullish", "reward", "win", "high", "growth", "approved", "boost",
    }
    NEGATIVE_WORDS = {
        "fail", "drop", "collapse", "loss", "decline", "crash", "vulnerability", "exploit",
        "impede", "deprivation", "hazard", "threat", "anticompetitive", "penalize", "chatter",
        "repetition", "persists", "strayed", "regret", "deficit", "bottleneck", "frozen",
        "toxic", "microplastics", "bearish", "margin", "risk", "warning", "flaw", "decay",
    }
    INTENSIFIERS = {"very", "major", "hugely", "significantly", "extremely", "substantially", "massive"}
    NEGATORS = {"not", "no", "never", "without", "none", "hardly", "rarely", "neither"}

    @classmethod
    def clean_text(cls, text: Optional[str]) -> str:
        """
        Cleans raw social media text:
        - Decodes HTML entities (&amp; -> &)
        - Strips URLs and hyperlinked Markdown syntax
        - Removes Reddit markdown bold, italic, spoilers, inline code
        - Normalizes whitespace
        """
        if text is None:
            return ""

        text = str(text)

        # 1. Unescape HTML entities
        text = html.unescape(text)

        # 2. Convert markdown links [anchor text](http://url) -> anchor text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        # 3. Remove raw URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # 4. Remove Reddit spoilers >!spoiler!< and markdown formatting
        text = re.sub(r">!(.*?)!<", r"\1", text)
        text = re.sub(r"[\*_~`]", "", text)

        # 5. Clean Reddit subreddit/user mentions (optional standardization)
        text = re.sub(r"\b[rRuU]/[a-zA-Z0-9_-]+", "", text)

        # 6. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def extract_linguistic_features(cls, title: str, body: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts structural and linguistic signals from title and body text.
        """
        cleaned_title = cls.clean_text(title)
        cleaned_body = cls.clean_text(body or "")
        combined = f"{cleaned_title} {cleaned_body}".strip()

        words = re.findall(r"\b\w+\b", combined)
        title_words = re.findall(r"\b\w+\b", cleaned_title)

        char_count = len(cleaned_title)
        word_count = len(title_words)

        # Uppercase ratio (shouting / hype index)
        uppercase_chars = sum(1 for c in cleaned_title if c.isupper())
        total_letters = sum(1 for c in cleaned_title if c.isalpha())
        uppercase_ratio = (uppercase_chars / total_letters) if total_letters > 0 else 0.0

        # Punctuation signals
        has_question = "?" in cleaned_title
        has_exclamation = "!" in cleaned_title
        question_count = cleaned_title.count("?")
        exclamation_count = cleaned_title.count("!")
        digit_count = sum(1 for c in cleaned_title if c.isdigit())
        has_numbers = digit_count > 0

        # Sentiment scoring
        sentiment_polarity, subjectivity, sentiment_label = cls.calculate_sentiment(combined)

        return {
            "cleaned_title": cleaned_title,
            "cleaned_body": cleaned_body,
            "title_char_count": char_count,
            "title_word_count": word_count,
            "uppercase_ratio": round(uppercase_ratio, 4),
            "has_question": int(has_question),
            "has_exclamation": int(has_exclamation),
            "question_count": question_count,
            "exclamation_count": exclamation_count,
            "has_numbers": int(has_numbers),
            "sentiment_polarity": sentiment_polarity,
            "sentiment_subjectivity": subjectivity,
            "sentiment_label": sentiment_label,
        }

    @classmethod
    def calculate_sentiment(cls, text: str) -> Tuple[float, float, str]:
        """
        Calculates sentiment polarity (-1.0 to +1.0) and subjectivity (0.0 to 1.0).
        Uses a vectorized lexicon engine with contextual negation handling.
        """
        words = re.findall(r"\b[a-zA-Z-]+\b", text.lower())
        if not words:
            return 0.0, 0.0, "neutral"

        pos_score = 0.0
        neg_score = 0.0
        subjective_words_count = 0

        for i, word in enumerate(words):
            # Check for negation in preceding 2 tokens
            is_negated = False
            if i > 0 and words[i - 1] in cls.NEGATORS:
                is_negated = True
            elif i > 1 and words[i - 2] in cls.NEGATORS:
                is_negated = True

            # Check intensifier in preceding token
            multiplier = 1.5 if (i > 0 and words[i - 1] in cls.INTENSIFIERS) else 1.0

            if word in cls.POSITIVE_WORDS:
                subjective_words_count += 1
                if is_negated:
                    neg_score += 1.0 * multiplier
                else:
                    pos_score += 1.0 * multiplier
            elif word in cls.NEGATIVE_WORDS:
                subjective_words_count += 1
                if is_negated:
                    pos_score += 0.8 * multiplier
                else:
                    neg_score += 1.0 * multiplier

        total_sentiment_words = pos_score + neg_score
        if total_sentiment_words == 0:
            polarity = 0.0
        else:
            polarity = (pos_score - neg_score) / (pos_score + neg_score)

        subjectivity = min(1.0, (subjective_words_count + 1) / (len(words) * 0.2 + 1))

        if polarity >= 0.15:
            label = "positive"
        elif polarity <= -0.15:
            label = "negative"
        else:
            label = "neutral"

        return round(polarity, 4), round(subjectivity, 4), label
