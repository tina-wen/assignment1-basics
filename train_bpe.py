import regex as re
from collections import Counter,defaultdict
from multiprocessing import Pool
import time
import sys
import json
from typing import Iterable, Iterator


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def merge(counts: dict[tuple[int,int],int], indices: list[int], pair: tuple[int, int], new_index: int, cnt: int):
    new_indices = []
    i,flag = 0,1 # flag=1表示前一指针指向 unmerged byte
    while i < len(indices):
        if i+1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
            # 更新indice
            new_indices.append(new_index)
            
            # 更新counts，只以前向做处理
            if i > 0:
                counts[(indices[i-1], indices[i])] -= cnt
            # 当前指针i指向merged byte
                if flag:
                # 前一指针指向unmerged byte
                    counts[(indices[i-1], new_index)] += cnt                    
                else:
                # 前一指针指向merged byte
                    counts[(new_index, new_index)] += cnt
            
            # 当前指针指向merged byte
            flag = 0
            i = i + 2
        else:
            # 更新indice
            new_indices.append(indices[i])
                
            # 更新counts
            if i > 0:
                if not flag:
                    counts[(indices[i-1], indices[i])] -= cnt
                    counts[(new_index, indices[i])] += cnt
            
            # 当前指针指向 unmerged byte
            flag = 1
            i += 1
    
    return new_indices
    
def word_split(word_cnt: dict[str, int], merge_times: int):

    # 初始化
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    word_idx: dict[str, list] = {}
        
    # 第一轮BPE训练
    counts = defaultdict(int)
    for word, cnt in word_cnt.items():
        # 对于每个word， 应该有一个indices，后续迭代训练可以只修改该indices
        indices = list(map(int, word.encode("utf-8")))
        word_idx[word] = indices
        # 统计连续二元bytes组的出现频率
        for index1, index2 in zip(indices, indices[1:]):
            counts[(index1, index2)] += cnt


    for _ in range(merge_times):
        # 统计完所有二元bytes出现次数后，更新merges和vocab：把出现频率最高的那个组加进来
        if not counts:
            break
        # 找到出现频率最高，且字典序最大的连续二元bytes组
        max_value = max(counts.values())
        max_keys = [(idxs, (vocab[idxs[0]],vocab[idxs[1]])) for idxs,val in counts.items() if val == max_value]
        pair,sep = sorted(max_keys, key = lambda x: x[1], reverse = True)[0]

        # 更新vocab和merges
        merges.append(sep)
        vocab[len(vocab)] = sep[0] + sep[1]
            
        # 更新word_idx和counts
        counts[pair] = 0
        for word, idx in word_idx.items():
            word_idx[word] = merge(counts, idx, pair, len(vocab)-1, word_cnt[word])
        
    return vocab, merges

def process_chunk(chunk):
    return Counter(re.findall(PAT, chunk))

def pre_tokenization(text: str, special_tokens: list[str] | None = None) -> Counter:
    num_processes = 4
    
    chunks = re.split("|".join(re.escape(sp_tok) for sp_tok in special_tokens), text)    
    ## 并行处理chunks
    # 人为切分chunks
    if len(chunks) == 1:
        origin_text = chunks[0]
        proc_text = re.findall(PAT, origin_text)
        chunk_size = len(proc_text) // num_processes
        boundaries = [i * chunk_size for i in range(num_processes + 1)]
        boundaries[-1] = len(proc_text)
        chunks = [proc_text[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])]
        with Pool(processes=num_processes) as pool:
            counters = pool.map(Counter, chunks)
    else:
        if len(chunks) < num_processes:
            num_processes = len(chunks)
        with Pool(processes=num_processes) as pool:
            counters = pool.map(process_chunk, chunks)

    word_cnt = Counter()
    for counter in counters:
        word_cnt.update(counter)
    return word_cnt

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    

    merge_times = vocab_size - len(special_tokens) - 256

    with open(input_path, "r", encoding ='utf-8') as f:
        text = f.read()#.decode("utf-8", errors="ignore")
    
    word_cnt = pre_tokenization(text, special_tokens)     
    vocab, merges = word_split(word_cnt, merge_times)

    for sp_tok in special_tokens:
        vocab[len(vocab)] = sp_tok.encode('utf-8')

    return vocab, merges
                
class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges

        # 同一个special token的合并
        self.special_tokens = special_tokens
                
        vocab_val = vocab.values()
        if special_tokens:
            for sp_tok in special_tokens:
                sp_byte = sp_tok.encode('utf-8')
                if sp_byte not in vocab_val:
                    vocab[len(vocab)] = sp_byte

        self.byte_encoder = {v: k for k, v in vocab.items()}
        # self.merge_dict = {pair: idx for idx, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # json格式储存字典：vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        vocab = {v: bytes(k.encode('utf-8')) for k, v in vocab_data.items()}

        # txt格式储存列表：merges
        merges = []
        with open(merges_filepath, 'r', encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) == 2:
                    byte1 = parts[0].encode('utf-8')
                    byte2 = parts[1].encode('utf-8')
                    merges.append((byte1,byte2))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # pre tokenization
        # tokens_list：原始text粗略拆出来的预分词单元（可能有重复）
        if self.special_tokens:
            chunks = re.split("("+ "|".join(re.escape(sp_tok) for sp_tok in sorted(self.special_tokens, key = len, reverse = True)) +")", text)
            chunks = [chunk for chunk in chunks if chunk != ' ']
        else:
            chunks = [text]
        tokens_list, no_special_tokens = [],[]
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens_list.append(chunk)
            else:
                tokens_list += re.findall(PAT, chunk)
                no_special_tokens += re.findall(PAT, chunk)
        
        # 哈希表的形式逐个处理预分词单元 
        initial_dict = {word: [bytes([integer]) for integer in word.encode('utf-8')] for word in set(no_special_tokens)}

        # 遍历merges
        for iter in self.merges:
            for word, byte_list in initial_dict.items():
                pos = 0
                new_bl = []
                while pos < len(byte_list):
                    if pos + 1 < len(byte_list) and (byte_list[pos],byte_list[pos + 1]) == iter:
                        new_bl.append(iter[0]+iter[1])
                        pos += 2
                    else:
                        new_bl.append(byte_list[pos])
                        pos += 1
                initial_dict[word] = new_bl
        
        # 根据哈希表，将bytes映射会原始text
        result = []
        for word in tokens_list:
            bl = initial_dict.get(word, [word.encode('utf-8')])
            result += [self.byte_encoder[single_byte] for single_byte in bl]
        return result

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            tokens = self.encode(text_chunk)
            for token in tokens:
                yield token

    def decode(self, ids: list[int]) -> str:
        # 注意，这个地方给定的vocab和训练的vocab构建方式不一致：训练的vocab前256个一定是单字节，中间是合并后的字节，最后是special tokens
        bytes_out = bytes([])
        for id in ids:
            bytes_out += self.vocab[id]
        return bytes_out.decode("utf-8",errors = 'replace')


