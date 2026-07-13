import wandb
from transformer import TransLM
from optim import AdamW, CosineScheduler
from loss import CrossEntropy
from utils import load_checkpoint, get_batch, save_checkpoint, GradClip

import numpy as np
import torch

def train(cfg):
    model = TransLM(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.vocab_size, cfg.num_layers, cfg.device)
    optimizer = AdamW(model.parameters(), cfg.lr, cfg.weight_decay, cfg.betas, cfg.eps)
    scheduler = CosineScheduler(optimizer, cfg.lr_max, cfg.lr_min, cfg.T_w, cfg.T_c)

    inputs = np.memmap(cfg.data_path + '/' + 'train.bin', dtype=np.uint16, mode='r')

    if cfg.resume:
        step = load_checkpoint(cfg.load_ckpt_path, model, optimizer)
    else:
        step = 0

    run = wandb.init(project="ass1", )

    torch.autograd.set_detect_anomaly(True)

    while step < cfg.max_train_steps:
        optimizer.zero_grad()

        x,y = get_batch(inputs,cfg.batch_size,cfg.context_length, device=cfg.device)
        outputs = model(x, cfg.context_length, cfg.rope_theta)
        
        loss = CrossEntropy(outputs,y)

        loss.backward()

        
        grad_norm = GradClip(model.parameters(),cfg.max_l2_norm,)

        if cfg.ckpt_save_step is not None and step == cfg.ckpt_save_step:
            save_checkpoint(model,optimizer,step,cfg.ckpt_save_path)

        optimizer.step()
        scheduler.step(step)

        wandb.log({
            "step": step,
            "train_loss": loss,
            "grad": grad_norm,
            "lr": optimizer.param_groups[0]['lr'],
        })


class TrainConfig:
    def __init__(self, d_model, num_heads, d_ff, vocab_size, num_layers,
                 lr, weight_decay, betas, eps,
                 lr_max, lr_min, T_w, T_c,
                 batch_size, context_length,
                 max_l2_norm,
                 data_path,
                 resume=False,
                 load_ckpt_path=None,
                 ckpt_save_step=None,
                 ckpt_save_path=None,
                 max_train_steps=100000):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_w = T_w
        self.T_c = T_c

        self.batch_size = batch_size
        self.context_length = context_length
        self.rope_theta = 10000

        self.max_l2_norm = max_l2_norm

        self.data_path = data_path

        self.resume = resume
        self.load_ckpt_path = load_ckpt_path
        self.ckpt_save_step = ckpt_save_step
        self.ckpt_save_path = ckpt_save_path

        self.max_train_steps = max_train_steps

        self.device = 'cuda'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    import json
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    
    # from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
    # tokenizer = get_tokenizer_from_vocab_merges_path('tests/fixtures/train-bpe-reference-vocab.json', 'tests/fixtures/train-bpe-reference-merges.txt', special_tokens = ["<|endoftext|>"])

    # with open("./tests/fixtures/tinystories_sample_5M.txt") as f:
    #     corpus_contents = f.read()

    # corpus_token_id = tokenizer.encode(corpus_contents)
    # tokens = np.array(corpus_token_id, dtype = np.uint16)

    # tokens.tofile('./data/train.bin')

    cfg = TrainConfig(**config_dict)
    train(cfg)