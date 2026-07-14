import torch
from loss import LogProb
import numpy as np
from train_bpe import BPETokenizer
from transformer import TransLM
from utils import load_checkpoint
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
tokenizer = get_tokenizer_from_vocab_merges_path('tests/fixtures/gpt2_vocab.json', 'tests/fixtures/gpt2_merges.txt', special_tokens = ["<|endoftext|>"])  


def top_p_sample(logits: torch.Tensor, p: float = 0.9, tao: float = 1.0):
    probs = torch.exp(LogProb(logits,tao)).squeeze() # V

    sorted_probs, sorted_indices = torch.sort(probs, descending = True)
    
    cumsum = torch.cumsum(sorted_probs, dim = 0)
    mask = cumsum >= p
    cutoff = torch.argmax(mask.int()) # 第一个最大值（mask的值不是0就是1）的索引，累积误差首次超过0.9的位置

    selected = sorted_indices[:cutoff+1]
    filtered_probs = torch.zeros_like(probs)
    filtered_probs[selected] = sorted_probs[:cutoff+1]
    filtered_probs = filtered_probs/filtered_probs.sum()
    return torch.multinomial(filtered_probs, 1).item()

def generate(x: torch.Tensor, model, tao: float = 1.0):
    seq_len = x.shape[1] # 1,S
    with torch.no_grad():
        logits = model(x, seq_len, rope_theta = 10000) # 1,S,V
    next_token = top_p_sample(logits[:, -1, :], tao = tao)
    return next_token


def inference(prompt: str, tokenizer: BPETokenizer, model: TransLM, stop_token: int, max_len: int, device: str):
    token_ids = tokenizer.encode(prompt)
    step = 0
    while step < max_len:
        x = np.array(token_ids, dtype = np.uint16)
        x = torch.from_numpy(x.astype(np.int64)).to(device)

        next_token = generate(x.unsqueeze(0),model)
        
        token_ids.append(next_token)
        if next_token == stop_token:
            return token_ids
        step += 1
    return token_ids


if __name__ == "__main__":
    model = TransLM(512, 16, 1344, 50257, 4, 'cuda')
    load_checkpoint('./ckpt/step_10000.pt', model, None)
    prompt =  "Lily and Ben were playing in the park."
    token_ids = inference(prompt, tokenizer, model, stop_token=0, max_len=100, device='cuda')
    outputs = tokenizer.decode(token_ids)
    print(f"Prompt:{prompt}\nOutput: {outputs}\n")