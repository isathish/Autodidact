"""
Phase 2, Milestone M2.3 — Unsupervised Token Discovery
Implements Byte-Pair Encoding (BPE) for learning tokens from raw text.
"""

from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.vocab = {}
        self.bpe_codes = {}

    def get_stats(self, corpus):
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, corpus):
        new_corpus = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in corpus:
            new_word = word.replace(bigram, replacement)
            new_corpus[new_word] = corpus[word]
        return new_corpus

    def fit(self, text_list):
        # Initialize corpus
        corpus = defaultdict(int)
        for text in text_list:
            for word in text.strip().split():
                corpus[' '.join(list(word)) + ' </w>'] += 1

        for i in range(self.num_merges):
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            corpus = self.merge_vocab(best, corpus)
            self.bpe_codes[best] = i

        self.vocab = corpus

    def encode(self, word):
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            if not pairs:
                break
            bigram = min(pairs, key=lambda pair: self.bpe_codes.get(pair, float('inf')))
            if bigram not in self.bpe_codes:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def decode(self, tokens):
        return ''.join(tokens).replace('</w>', '')


if __name__ == "__main__":
    tokenizer = BPETokenizer(num_merges=50)
    tokenizer.fit(["this is a test", "this test is fun"])
    print("Encoded:", tokenizer.encode("test"))
    print("Decoded:", tokenizer.decode(tokenizer.encode("test")))
