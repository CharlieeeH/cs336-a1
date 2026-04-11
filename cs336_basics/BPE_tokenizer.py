import os
from collections import defaultdict, Counter
import regex as re
import json

special_tokens = []


def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, trains a (byte-level) BPE tokenizer.
    """

    # --- 1. Initialize basic vocabulary ---
    vocab = {i: bytes([i]) for i in range(256)}

    num_merges = vocab_size - 256 - len(special_tokens)

    # --- 2. read corpus and segment by special_tokens ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

        if special_tokens:
            special_regex = "|".join(re.escape(t) for t in special_tokens)
            parts = re.split(f"({special_regex})", text)
            train_segments = [p for p in parts if p not in special_tokens]
        else:
            train_segments = [text]


    # --- 3. Pretokenization ---
    # Code data structures:  words_list, counts_list, stats, indices
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    raw_counts = Counter()
    for segment in train_segments:
        words = gpt2_pat.findall(segment)
        for word in words:
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1

    words_list = []
    counts_list = []

    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)

    stats = defaultdict(int)    # stats is a dictionary with default init values
    indices = defaultdict(set)   # inverted index
    
    for idx, word in enumerate(words_list):
        freq = counts_list[idx]
        for i in range(len(word) -1):
            pair = (word[i], word[i+1])
            stats[pair] += freq
            indices[pair].add(idx)
    
    merges = []

    # --- 4. iterations of merging ---

    for _ in range(num_merges):
        if not stats:
            break

        best_pair = max(stats.items(), key = lambda x: (x[1], x[0]))[0]

        if stats[best_pair] <= 0:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        
        # now find those words that need update
        # we need to modify words_list, stats and indices
        relevant_indices = list(indices[best_pair])

        for idx in relevant_indices:
            word = words_list[idx]
            freq = counts_list[idx]

            i = 0
            while i < len(word) -1:
                if word[i]==best_pair[0] and word[i+1]==best_pair[1]:
                    if i>0:
                        prev_pair = (word[i-1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                            # del indices[prev_pair]
                    if i<len(word)-2:
                        next_pair = (word[i+1], word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] ==0:
                            del stats[next_pair]
                            # del indices[next_pair]

                    word[i] = new_token
                    del word[i+1]

                    if i>0:
                        new_prev = (word[i-1], word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)
                    if i<len(word)-1:  # here we cannot use -2 because len(word) has been decreased by 1
                        new_next = (word[i], word[i+1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)
                else: 
                    i+=1
        
        # clean remaining data relevant to best_pair
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]

    # finally update vocab using merges and special_tokens    
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0]+pair[1]

    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[len(vocab)] = s_bytes

    return vocab, merges


def train_bpe_tinystories():
    pass


def train_bpe_expts_owt():
    pass


# class Tokenizer:
#     def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
#         pass

#     def from_files(cls, vocab_filepath, merge_file_path, special_tokens=None):
#         pass

#     def encode(self, text: str) -> list[int]:
#         pass

#     def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
#         pass

#     def decode(self, ids: list[int]) -> str:
#         pass
