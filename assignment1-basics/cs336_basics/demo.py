import json
import pathlib
import re
import time
from bpe_tokenizer import BPETokenizer

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"

# input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
# vocab_size = 10000
# special_tokens = ["<|endoftext|>"]

# vocab, merges = BPETokenizer.train(input_path, vocab_size, special_tokens)
# begin_time = time.time()
# vocab = {key: value.decode("utf-8") for key, value in vocab.items()}
# merges = [v.decode("utf-8") for v in merges]

# print(f"cost: {time.time() - begin_time}")

# with open(f"{DATA_PATH}/vocab.json", "w") as fp:
#     json.dump(vocab, fp, indent=2)
# with open(f"{DATA_PATH}/merges.json", "w") as fp:
#     json.dump(merges, fp, indent=2)
g = {"b": 1}


def test(dd: dict):
    x = dd
    x["a"] = 1


test(g)
print(g)
