import os
from typing import List
from collections import defaultdict
import regex as re
import heapq

# from heapdict import heapdict
from multiprocessing import Pool
from collections import Counter
from typing import BinaryIO
from typing import Iterable, Iterator, List, Dict, Tuple
import copy
import pickle
import time
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def merge(tokens, pair, index):
    i = 0
    no_of_appearances = 0
    merged_tokens = []
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged_tokens.append(index)
            no_of_appearances += 1
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1
    # if no_of_appearances > 1:
    #     print(pair, tokens, no_of_appearances)
    return merged_tokens, no_of_appearances


def get_special_tokens_regex(special_tokens):
    if not special_tokens:
        return None
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))

    pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
    return re.compile(f'({pattern})')


def pre_tokenize(texts: str, special_tokens: list, special_token_pattern: str):
    if special_tokens:
        parts = special_token_pattern.split(texts)
    else:
        parts = [texts]
        
    word_count = Counter()

    for part in parts:
        
        if special_tokens and part in special_tokens:
            # print("skipping special token", part)
            continue
        # pre_tokenized_text = re.finditer(PAT, part)
        # tokens = re.findall(PAT, part)
        tokens = PAT.findall(part)
        word_count.update(tokens)

    return word_count


def process_chunk(args):
    chunk, special_tokens, special_token_pattern = args
    word_count = pre_tokenize(chunk, special_tokens, special_token_pattern)
    return word_count


def count_words_parallel(chunks, special_tokens, workers=4):
    special_token_pattern = get_special_tokens_regex(special_tokens)
    args = [(chunk, special_tokens, special_token_pattern) for chunk in chunks]

    with Pool(workers) as pool:
        results = pool.map(process_chunk, args)

    # merge results
    total_counts = Counter()
    for wc in results:
        total_counts.update(wc)

    return total_counts


def train_bpe(input_path, vocab_size, special_tokens):
    start_time = time.time()
    no_of_workers = 12
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, no_of_workers, "<|endoftext|>".encode("utf-8"))
        chunks = []
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")
            chunks.append(chunk)
    print(f"chunking done in {time.time() - start_time} seconds.")

    word_counts = count_words_parallel(chunks=chunks, special_tokens=special_tokens, workers=no_of_workers)
    print(f"Pre-tokenization done in {time.time() - start_time} seconds.")

    bytes_count = {}
    for word, count in word_counts.items():
        word_bytes = tuple(word.encode("utf-8"))
        bytes_count[word_bytes] = count

    pair_counts = {}
    pair_words_mapping = {}
    bytes_mapping = {}
    bytes_count_mapping = {}
    for i, (tokens, count) in enumerate(bytes_count.items()):
        bytes_mapping[i] = tokens
        bytes_count_mapping[i] = count
        for pair in zip(tokens[:-1], tokens[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_words = pair_words_mapping.get(pair, None)
            if not pair_words:
                pair_words_mapping[pair] = [i]
            else:
                pair_words_mapping[pair].append(i)

    merges = []

    vocab = {x: bytes([x]) for x in range(0, 256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    next_index = len(vocab)

    while len(vocab) != vocab_size and len(pair_counts) > 0:
        max_value = max(pair_counts.values())

        max_keys = []
        decoded_max_keys = []
        for k, v in pair_counts.items():
            if v == max_value:
                max_keys.append(k)
                decoded_max_keys.append(
                    (vocab[k[0]].decode("utf-8", errors="replace"), vocab[k[1]].decode("utf-8", errors="replace"))
                )

        highest_pair = max(decoded_max_keys)
        highest_pair = max_keys[decoded_max_keys.index(highest_pair)]

        token1_bytes = vocab[highest_pair[0]]
        token2_bytes = vocab[highest_pair[1]]
        merges.append((token1_bytes, token2_bytes))
        new_token_bytes = token1_bytes + token2_bytes
        vocab[next_index] = new_token_bytes

        affected_word_indices = list(pair_words_mapping.get(highest_pair, []))

        for word_idx in affected_word_indices:
            word_tokens = bytes_mapping[word_idx]
            count = bytes_count_mapping[word_idx]

            old_pairs = list(zip(word_tokens, word_tokens[1:]))

            new_word_tokens, no_of_appearances = merge(word_tokens, highest_pair, next_index)
            bytes_mapping[word_idx] = new_word_tokens

            new_pairs = list(zip(new_word_tokens, new_word_tokens[1:]))
            for pair in old_pairs:  # decrease counts for pair containing one of the merged tokens
                if pair in new_pairs:
                    # Removes duplicate pairs. eg for (a, b, c, a, b) with (b,c) as merge pairs, (a,b) appears twice.
                    # after merging, (a,b) appears only once. so it will make sure it only decreases the count as many times as it appears
                    new_pairs.remove(pair)
                    continue
                old_priority = pair_counts.get(pair)
                if old_priority:
                    new_count = old_priority - count
                    pair_counts[pair] = new_count

                    if new_count <= 0:
                        pair_counts.pop(pair)

                if pair in pair_words_mapping:
                    pair_words_mapping[pair].remove(word_idx)
                    if not pair_words_mapping[pair]:
                        del pair_words_mapping[pair]

            for pair in new_pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                if pair not in pair_words_mapping:
                    pair_words_mapping[pair] = []
                pair_words_mapping[pair].append(word_idx)

        next_index += 1

    return vocab, merges


class BPETokenizer:
    """
    A BPE tokenizer that uses a vocabulary, a list of merges, and a list of special tokens.
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        # self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.special_token_pattern = None
        if special_tokens:
            self.special_token_pattern = get_special_tokens_regex(self.special_tokens)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.replacement_byte = bytes("\uFFFD", encoding="utf-8")
        
    
    def encode(self, text:str)->list[int]:
        if self.special_tokens:
            parts = self.special_token_pattern.split(text)
        else:
            parts = [text]
        
        pre_tokens = []
        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                pre_tokens.append(part)
            else:
                found_pre_tokens =PAT.findall(part)
                if found_pre_tokens:
                    pre_tokens.extend(found_pre_tokens)

        char_bytes = []
        for token in pre_tokens:
            if self.special_tokens and token in self.special_tokens:
                char_bytes.append([self.reverse_vocab[token.encode("utf-8")]])
            else:
                char_list = [self.reverse_vocab[bytes([b])] for b in token.encode("utf-8")]
                char_bytes.append(char_list)
    
        pre_tokens = []
        
        for i, token in enumerate(char_bytes):
            while True:
                token_len =len(token)
                if token_len == 1:
                    break
                
                low_rank_pair = None
                rank = None
                for pair in zip(token[:-1], token[1:]):
                    byte_pair = (self.vocab[pair[0]], self.vocab[pair[1]])
                    # print(pair, byte_pair)
                    cur_rank = self.merge_ranks.get(byte_pair, None)
                    if cur_rank is not None:
                        if rank is None:
                            rank = cur_rank
                            low_rank_pair = pair
                        elif cur_rank < rank:
                            rank = cur_rank
                            low_rank_pair = pair
                
                if rank is None:
                    break
                
                low_rank_pair_bytes = (self.vocab[low_rank_pair[0]], self.vocab[low_rank_pair[1]])
                
                temp_token = []
                j = 0
                while j < token_len:
                    if j + 1 < token_len and token[j] == low_rank_pair[0] and token[j + 1] == low_rank_pair[1]:
                        merged_byte = low_rank_pair_bytes[0] + low_rank_pair_bytes[1]
                        merged_token = self.reverse_vocab[merged_byte]
                        temp_token.append(merged_token)
                        j += 2
                    else:
                        temp_token.append(token[j])
                        j += 1
                token = temp_token
            pre_tokens.append(token)
                
        # print(pre_tokens)
        tokens = []
        for pre_token in pre_tokens:
            tokens.extend(pre_token)
            # for token in pre_token:
            #     tokens.append(self.vocab[bytes([token])])
        return tokens            

    def decode(self, ids:list[int])->str:
        decoded_bytes = bytes()
        
        for token_id in ids:
            if token_id in self.vocab:
                decoded_bytes += self.vocab[token_id]
            else:
                print(token_id)
                decoded_bytes += self.replacement_byte
        return decoded_bytes.decode(encoding="utf-8", errors="replace")

    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def encode_iterable(self, iterable: Iterable[str])->Iterator[int]:
        for text in iterable:
            yield from self.encode(text)


if __name__ == "__main__":
    from pathlib import Path
    from cProfile import Profile
    from pstats import SortKey, Stats
    
    vocab_size = 10000
    file_path = "/home/fm-pc-lt-270/Documents/stanford/first/data/TinyStoriesV2-GPT4-train.txt"
    # file_path = "/home/fm-pc-lt-270/Documents/stanford/first/data/TinyStoriesV2-GPT4-valid.txt"
    # file_path = "/home/fm-pc-lt-270/Documents/stanford/first/data/test.txt"

    # special_tokens = ["<|endoftext|>"]
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    # special_tokens = None
    start_time = time.time()
    # vocab, merges = train_bpe(file_path, vocab_size, special_tokens)
    # with open("tiny_stories_vocab.pkl", "wb") as f:
    #     pickle.dump(vocab, f)
    # with open("tiny_stories_merges.pkl", "wb") as f:
    #     pickle.dump(merges, f)

    
    with open("tiny_stories_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("tiny_stories_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    end_time = time.time()
    print("time:", end_time - start_time)
    # print(merges)
    # print(vocab)
    tokenizer = BPETokenizer(vocab, merges, special_tokens)

    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # test_string = "🙃"
    # test_string = "s"
#     test_string = """
# Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed!
# He said, “Wow, that is a really amazing vase! Can I buy it?”
# The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!”
# So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was.
# And that's how Ben found an amazing vase in the store!
# <|endoftext|>
# Once upon a time, there was a reliable otter named Ollie. He lived in a river with his family. They all loved to play and swim together.
# One day, Ollie's mom said, "Ollie, hurry and get some fish for dinner!" Ollie swam fast to catch fish. He saw his friend, the duck. "Hi, Ollie!" said the duck. "Hi, duck!" said Ollie. "I need to hurry and catch fish for my family."
# While Ollie was catching fish, he found a big shiny stone. He thought, "This is not a fish, but it is so pretty!" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner.
# <|endoftext|>
# One day, a little boy named Tim went to the park. He saw a big tiger. The tiger was not mean, but very easy to play with. Tim and the tiger played all day. They had lots of fun.
# Then, something unexpected happened. The tiger started to shake. Tim was scared. He did not know what was going on. But then, the tiger turned into a nice dog. Tim was very surprised.
# Tim and the dog played together now. They were very happy. The dog was easy to play with too. At the end of the day, Tim went home with his new friend.
# <|endoftext|>

# Once upon a time there was a friendly little boy called Bob. Bob loved to pick flowers and look for birds. One day he decided to go outside with his friends to pick some more flowers.
# He suddenly noticed something weird on the ground. It was a big, green thumb! It was so big, Bob had never seen one before. Bob curiously leaned in to take a better look. He told his friends: "look everyone, I picked up this big thumb! What do we do with it?"
# His friends were very excited. They told him to pick it up and take it home to show his family. So Bob carefully picked up the friendly thumb and carried it back home. When he arrived, Bob happily showed the thumb to his family. His dad was amazed and hugged Bob to show his appreciation.
# From that day on Bob always kept the big, friendly thumb with him as a reminder that special things can be found anywhere.
# <|endoftext|>
# Once upon a time, in a small house, there lived a little girl named Lucy. Lucy loved the color orange. She had an orange dress, an orange ball, and even an orange cat. One day, Lucy met a new friend. This friend was not like other friends. It was a spirit. The spirit was very nice and liked to play with Lucy.
# One day, Lucy and the spirit were playing with her orange ball. They were having so much fun. Then, Lucy's mom called her for dinner. Lucy said to the spirit, "I have to go eat now. Will you play with me later?" The spirit nodded and smiled.
# At dinner, Lucy told her mom about the spirit. But her mom did not believe her. She said, "Spirits are not real, Lucy. You have a big imagination." Lucy felt sad that her mom did not believe her. After dinner, she went back to play with the spirit. They played with the orange ball and had lots of fun. Lucy knew that even if others ignore her friend, the spirit was real and they could play together.
# <|endoftext|>
# """
    with Profile() as profile:
        encoded = tokenizer.encode(test_string)
    Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()
    
    print("encoded:", encoded)
    decoded = [tokenizer.decode([x]) for x in encoded]
    print("decoded:", decoded)
    decoded = tokenizer.decode(encoded)
    print("decoded:", decoded)

    # # assert(decoded == test_string)
    # print(test_string == decoded)
    # print(len(vocab), len(merges))
    # import pickle

    # with open('/home/fm-pc-lt-270/Documents/stanford/stanford-cs336-a1/vocab.pkl', 'rb') as f:
    #     ref_vocab = pickle.load(f)

    # with open('/home/fm-pc-lt-270/Documents/stanford/stanford-cs336-a1/merges.pkl', 'rb') as f:
    #     ref_merges = pickle.load(f)

    # # print(vocab == ref_vocab)
    # # print(merges == ref_merges)

    # error_merges = 0
    # for i, merge, ref_merge in zip(range(len(merges)),merges, ref_merges):
    #     if merge != ref_merge:
    #         error_merges += 1
    #         print(i, merge, ref_merge)

    # print(error_merges)
