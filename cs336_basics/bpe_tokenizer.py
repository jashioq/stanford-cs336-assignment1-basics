from concurrent.futures import ProcessPoolExecutor
from itertools import islice, pairwise, repeat
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

        max_pair = max(candidates)
        joined = b''.join(max_pair)

        vocab[next_token_id] = joined
        next_token_id += 1

        merges.append(max_pair)

        # replace merged pairs here in groups

        keys = [groups.keys()]
        for k in keys:
            if len(k) > 1:
                for f, s in pairwise(k):
                    j =  b''.join([f, s])
                    if j == joined:
                        #replace

        # print(merges)
        # print(vocab.items())
        





    

    # for k, v in islice(counts.items(), 10):
    #     print(k, v)


    









def pretokenize(input_path: str, boundries: tuple[int, int], special_tokens: list[str]):
    start, end = boundries

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    # Remove special tokens
    for st in special_tokens:
        chunk = chunk.replace(st, "")


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
        b = tuple([char.encode("utf-8") for char in m])
        counts[b] = counts.get(b, 0) + 1

    print("process finished pretokenizing a chunk")

    return counts










if __name__ == "__main__":
    train("data/TinyStoriesV2-GPT4-valid.txt", 258, ["<|endoftext|>"])