from concurrent.futures import ProcessPoolExecutor
from itertools import islice, pairwise, repeat
import os
import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

dummy = " low low low low low lower lower widest widest widest newest newest newest newest newest newest"

def run(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "rb") as file:
        # Split for parallel pre-tokenization
        num_processes = os.cpu_count() or 1
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

        print(f"starting pretokezning with {num_processes} processes")
        print(f"chunk boundries: {boundaries}")


    with ProcessPoolExecutor() as ex:
        dicts = list(ex.map(pretokenize, repeat(input_path), pairwise(boundaries), repeat(special_tokens)))

    groups = {}
    for d in dicts:
        for k, v in d.items():
            groups[k] = groups.get(k, 0) + v

    vocab = {i: bytes([i]) for i in range(256)}

    next_token_id = 256
    for st in special_tokens:
        vocab[next_token_id] = st.encode("utf-8")
        next_token_id += 1

    merges = []

    for i in range(vocab_size - len(special_tokens) - 256):
        counts = {}
        for k, v in groups.items():
            for p in pairwise(k):
                counts[p] = counts.get(p, 0) + v

        counts = {key: value for key, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)}

        max_occurance = counts.get(next(iter(counts.keys())))
        candidates = []
        for k, v in counts.items():
            if v == max_occurance:
                candidates.append(k)
            else:
                break

        max_pair = tuple(max(candidates))
        joined = b''.join(max_pair)

        vocab[next_token_id] = joined
        next_token_id += 1

        merges.append(max_pair)

        # replace merged pairs here in groups

        new_groups = {}
        for k, count in groups.items():
            new_k = merge_pair(k, max_pair, joined)
            new_groups[new_k] = new_groups.get(new_k, 0) + count
        groups = new_groups

    return vocab, merges
        

def merge_pair(key, pair, joined):
    out = []
    i = 0
    while i < len(key):
        if i + 1 < len(key) and key[i] == pair[0] and key[i + 1] == pair[1]:
            out.append(joined)
            i += 2
        else:
            out.append(key[i])
            i += 1
    return tuple(out)


def pretokenize(input_path: str, boundries: tuple[int, int], special_tokens: list[str]):
    start, end = boundries

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    # Remove special tokens
    if special_tokens:
        chunks = re.split("|".join(re.escape(st) for st in special_tokens), chunk)
    else:
        chunks = [chunk]

    matches = []
    for c in chunks:  
        pre_token_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches.extend([m.group() for m in re.finditer(pre_token_regex, c)])

    counts = {}
    for m in matches:
        encoded = m.encode("utf-8")
        b = tuple([encoded[i:i+1] for i in range(len(encoded))])
        counts[b] = counts.get(b, 0) + 1

    print("process finished pretokenizing a chunk")

    return counts


def train(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return run(input_path, vocab_size, special_tokens)