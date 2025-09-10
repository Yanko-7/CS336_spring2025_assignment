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
text = """
Once upon a time, there was a reliable otter named Ollie. He lived in a river with his family. They all loved to play and swim together.
One day, Ollie's mom said, "Ollie, hurry and get some fish for dinner!" Ollie swam fast to catch fish. He saw his friend, the duck. "Hi, Ollie!" said the duck. "Hi, duck!" said Ollie. "I need to hurry and catch fish for my family."
While Ollie was catching fish, he found a big shiny stone. He thought, "This is not a fish, but it is so pretty!" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner.
<|endoftext|>
One day, a little boy named Tim went to the park. He saw a big tiger. The tiger was not mean, but very easy to play with. Tim and the tiger played all day. They had lots of fun.
Then, something unexpected happened. The tiger started to shake. Tim was scared. He did not know what was going on. But then, the tiger turned into a nice dog. Tim was very surprised.
Tim and the dog played together now. They were very happy. The dog was easy to play with too. At the end of the day, Tim went home with his new friend.
<|endoftext|>
"""
special_tokens = ["<|endoftext|>"]
escaped = [re.escape(special_token) for special_token in special_tokens]
for str in re.split(f"({"|".join(escaped)})", text):
    print(str)
