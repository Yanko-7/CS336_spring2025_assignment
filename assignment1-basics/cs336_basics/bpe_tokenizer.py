import re, regex
import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class BPETokenizer:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, special_tokens):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.id = 255
        for special_token in special_tokens:
            self.id = self.id + 1
            self.vocab[self.id] = special_token.encode("utf-8")

    def merge_once(self, word_cnt_list: list) -> tuple[bytes, bytes]:
        pair_counts = {}
        pair2idx = {}
        for i in range(0, len(word_cnt_list)):
            bytes_list = word_cnt_list[i][0]
            cnt = word_cnt_list[i][1]
            for j in range(1, len(bytes_list)):
                pair = (bytes_list[j - 1], bytes_list[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + cnt
                pair2idx.setdefault(pair, []).append((i, j))
        max_pair = max(pair_counts, key=lambda p: (pair_counts[p], p[0], p[1]))

        need_merge_idx = {}
        for i, j in pair2idx[max_pair]:
            need_merge_idx.setdefault(i, []).append(j)

        for i, jl in need_merge_idx.items():
            selected = []
            last = -1
            for j in jl:
                if j - 1 > last:
                    selected.append(j)
                    last = j
            for j in reversed(selected):
                if (
                    j < len(word_cnt_list[i][0])
                    and (word_cnt_list[i][0][j - 1], word_cnt_list[i][0][j]) == max_pair
                ):
                    word_cnt_list[i][0][j - 1 : j + 1] = [max_pair[0] + max_pair[1]]

        return max_pair

    def train(
        self, input_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            # chunk_bd = zip(boundaries[:-1], boundaries[1:])
            total_words_dict = {}
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                for word, cnt in self.pre_tokenization(
                    f, start, end, special_tokens
                ).items():
                    total_words_dict[word] = total_words_dict.get(word, 0) + cnt

            iter_cnt = vocab_size - len(self.vocab)
            word_cnt_list = [
                [[bytes([b]) for b in word], cnt]
                for word, cnt in total_words_dict.items()
            ]
            merges = []
            for i in range(iter_cnt):
                merge_tuple = self.merge_once(word_cnt_list)
                merges.append(merge_tuple)
                self.id = self.id + 1
                self.vocab[self.id] = merge_tuple[0] + merge_tuple[1]

            return self.vocab, merges
        return (), []

    def pre_tokenization(self, f, start: int, end: int, special_tokens: list[str]):
        f.seek(start)
        words_dict = {}
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        escaped = [re.escape(special_token) for special_token in special_tokens]
        for str in re.split("|".join(escaped), chunk):
            iter = regex.finditer(self.PAT, str)
            for match in iter:
                word = match.group(0).encode("utf-8")
                words_dict[word] = words_dict.get(word, 0) + 1
        return words_dict
