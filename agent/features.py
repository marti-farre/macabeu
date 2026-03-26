"""Text feature extraction for RL defense selector state representation."""

import re
from collections import Counter
from typing import List

import numpy as np


class TextFeatureExtractor:
    """Extract statistical features from text for defense selection.

    Features capture signatures of different attack types:
    - DeepWordBug: high OOV ratio, char-level perturbations
    - BERTattack/PWWS/Genetic: word-level changes, near-normal char stats
    - Homoglyph attacks: high non-ASCII ratio
    """

    FEATURE_NAMES = [
        'text_length', 'word_count', 'avg_word_length',
        'oov_ratio', 'non_ascii_ratio', 'uppercase_ratio',
        'punctuation_ratio', 'digit_ratio', 'repeated_char_ratio',
        'char_entropy'
    ]
    NUM_FEATURES = len(FEATURE_NAMES)

    def __init__(self):
        self._vocab = None

    @property
    def vocab(self):
        """Load English vocabulary from SymSpell dictionary."""
        if self._vocab is None:
            try:
                from symspellpy import SymSpell
                import importlib.resources
                ss = SymSpell()
                dict_path = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
                ss.load_dictionary(str(dict_path), term_index=0, count_index=1)
                self._vocab = set(ss.words.keys())
            except (ImportError, Exception):
                self._vocab = set()
        return self._vocab

    def extract(self, text: str) -> np.ndarray:
        """Extract feature vector from a single text."""
        chars = list(text)
        words = text.split()
        n_chars = max(len(chars), 1)
        n_words = max(len(words), 1)

        features = np.zeros(self.NUM_FEATURES, dtype=np.float32)

        features[0] = min(n_chars / 2000.0, 1.0)
        features[1] = min(n_words / 400.0, 1.0)
        features[2] = np.mean([len(w) for w in words]) / 20.0 if words else 0.0

        # OOV ratio
        word_cores = [re.sub(r'[^a-zA-Z]', '', w.lower()) for w in words]
        word_cores = [w for w in word_cores if len(w) > 0]
        if word_cores and self.vocab:
            oov = sum(1 for w in word_cores if w not in self.vocab)
            features[3] = oov / len(word_cores)

        features[4] = sum(1 for c in chars if ord(c) > 127) / n_chars
        features[5] = sum(1 for c in chars if c.isupper()) / n_chars
        features[6] = sum(1 for c in chars if c in '.,!?;:"\'-()[]{}') / n_chars
        features[7] = sum(1 for c in chars if c.isdigit()) / n_chars

        repeated = sum(1 for i in range(1, len(chars)) if chars[i] == chars[i - 1])
        features[8] = repeated / n_chars

        counter = Counter(chars)
        probs = np.array(list(counter.values()), dtype=np.float32) / n_chars
        features[9] = -np.sum(probs * np.log2(probs + 1e-10)) / 8.0

        return features

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """Extract features for a batch of texts."""
        return np.array([self.extract(t) for t in texts])
