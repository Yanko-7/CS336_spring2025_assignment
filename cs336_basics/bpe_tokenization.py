from dataclasses import dataclass
import re,regex
from pretokenization_example import find_chunk_boundaries


class BPETokenizer:

    def __init__(self):
        self.vocab = {i.to_bytes(): i for i in range(256)}
        self.vocab[b"<|endoftext|>"] = 256
        pass

    def train(
        self, input_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                # TODO: parallel
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk = re.split(chunk, "|".join(special_tokens))
                regex.finditer(PAT,)
    def process_chunk(chunk: str, vocab_size: int):
        ()
