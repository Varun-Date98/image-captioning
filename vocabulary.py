import re
import string
from typing import List
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

class Vocabulary:
    def __init__(self, freq_threshold: int):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary_from_captions(self, captions_file: str, training_images: List[str]):
        idx = 4
        lines = None
        counter = Counter()

        # Reading the captions
        with open(captions_file, "r") as f:
            lines = f.readlines()

        # Tokenize each line
        for line in lines[1:]:
            if line.startswith("image"):
                print("Skipping line")
                continue

            img, caption = line.split(",", 1)

            if img not in training_images:
                continue

            caption = re.sub(f"[{re.escape(string.punctuation)}]", "", caption)
            tokens = word_tokenize(caption.lower())
            counter.update(tokens)

        # Building vocabulary
        for token, count in counter.items():
            if count >= self.freq_threshold:
                self.itos[idx] = token
                self.stoi[token] = idx
                idx += 1

        print(f"Vocabulary built successfully.")

    def numericalize(self, sentence: str) -> List[int]:
        # Tokenize sentence and return indexes
        sentence = re.sub(f"[{re.escape(string.punctuation)}]", "", sentence)
        tokens = word_tokenize(sentence.lower())
        tokens = ["<SOS>"] + tokens + ["<EOS>"]
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokens
        ]