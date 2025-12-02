"""
Chinese text pretokenization utilities.

This module provides functions for detecting and segmenting Chinese text using the jieba
library for word segmentation.
"""

from functools import cache

import regex


def has_thai(text: str) -> bool:
    """
    Check if the given text contains Thai characters.

    Args:
        text: The input text to check for Thai characters

    Returns:
        True if Thai characters are found, False otherwise
    """
    # Match any Han ideograph using Unicode property
    return bool(regex.search(r"[\u0E00-\u0E7F]", text))


@cache
def get_thai_segmenter():
    """
    Get a cached instance of the PyThaiNLP Thai word segmenter.

    PyThaiNLP is a popular Thai text processing library that uses a combination of
    dictionary-based matching and statistical models to segment Thai text into words.
    The segmenter is cached to avoid repeated initialization overhead.

    Returns:
        pythainlp module instance for text segmentation

    Raises:
        ImportError: If the PyThaiNLP library is not installed
    """
    try:
        import pythainlp.tokenize
    except ImportError:
        print("Error: pythainlp library not found. Please install it with: pip install pythainlp")
        raise

    return pythainlp.tokenize


def segment_thai(text: str) -> list[str]:
    """
    Segment Thai text into space-separated words.

    Args:
        text: The Thai text to segment

    Returns:
        List of Thai words

    Example:
        >>> segment_thai("สวัสดีโลก")
        "สวัสดี โลก"
    """
    pythainlp = get_thai_segmenter()
    # Use pythainlp.word_tokenize() for precise segmentation and join with spaces
    segments = pythainlp.word_tokenize(text)
    # Filter out empty segments and join with single spaces
    return list(segments)
