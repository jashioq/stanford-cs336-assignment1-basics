from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise, repeat
import os
import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

dummy = " low low low low low lower lower widest widest widest newest newest newest newest newest newest"

def train(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "rb") as file:
        # Split for parallel pre-tokenization
        num_processes = os.cpu_count() or 1
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

        print(f"starting pretokezning with {num_processes} processes")
        print(f"chunk boundries: {boundaries}")


    with ProcessPoolExecutor() as ex:
        dicts = list(ex.map(pretokenize, repeat(input_path), pairwise(boundaries), repeat(special_tokens)))

    counts = {}
    for d in dicts:
        for k, v in d.items():
            counts[k] = counts.get(k, 0) + v

    print(len(counts.keys()))
    print(len(set(counts.keys())))


def pretokenize(input_path: str, boundries: tuple[int, int], special_tokens: list[str]):
    start, end = boundries

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    # Remove special tokens
    for st in special_tokens:
        chunk = chunk.replace(st, "")

    # Pre-tokenization
    pre_token_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    matches = [m.group() for m in re.finditer(pre_token_regex, chunk)]

    counts = {}
    for m in matches:
        b = tuple([char.encode("utf-8") for char in m])
        counts[b] = counts.get(b, 0) + 1

    print("process finished pretokenizing a chunk")

    return counts










if __name__ == "__main__":
    train("data/TinyStoriesV2-GPT4-train.txt", 0, "<|endoftext|>")