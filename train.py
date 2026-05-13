"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))

            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

            # zero out pad positions
            true_dist[target == self.pad_idx] = 0

        loss = torch.sum(-true_dist * log_probs, dim=-1)

        # ignore pad positions in loss averaging
        non_pad_mask = target != self.pad_idx
        loss = loss[non_pad_mask].mean()

        return loss


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.
    """

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_tokens = 0

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        # 🔥 Target shifting (CRITICAL)
        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Masks
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        # Forward
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Flatten for loss
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        tgt_output = tgt_output.reshape(-1)

        loss = loss_fn(logits, tgt_output)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        # track loss
        non_pad = (tgt_output != loss_fn.pad_idx).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / total_tokens

# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).long().to(device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys)

        out = model.decode(memory, src_mask, ys, tgt_mask)

        next_token = out[:, -1, :].argmax(dim=-1).item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src).fill_(next_token)], dim=1
        )

        if next_token == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    """
    from evaluate import load
    bleu = load("bleu")

    model.eval()

    predictions = []
    references = []

    # helper for vocab lookup
    if isinstance(tgt_vocab, dict):
        itos = {v: k for k, v in tgt_vocab.items()}
    else:
        itos = tgt_vocab.itos
    
    def idx_to_token(idx):
        return itos.get(idx, "<unk>")

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            batch_size = src.size(0)

            for i in range(batch_size):
                src_i = src[i].unsqueeze(0)
                tgt_i = tgt[i]

                src_mask = make_src_mask(src_i)

                pred_tokens = greedy_decode(
                    model,
                    src_i,
                    src_mask,
                    max_len=max_len,
                    start_symbol=tgt_vocab["<sos>"] if isinstance(tgt_vocab, dict) else tgt_vocab.stoi["<sos>"],
                    end_symbol=tgt_vocab["<eos>"] if isinstance(tgt_vocab, dict) else tgt_vocab.stoi["<eos>"],
                    device=device,
                ).squeeze(0).tolist()

                # convert to tokens
                pred_sentence = [idx_to_token(idx) for idx in pred_tokens]
                tgt_sentence  = [idx_to_token(idx) for idx in tgt_i.tolist()]

                # remove special tokens
                pred_sentence = [tok for tok in pred_sentence if tok not in ["<sos>", "<eos>", "<pad>"]]
                tgt_sentence  = [tok for tok in tgt_sentence  if tok not in ["<sos>", "<eos>", "<pad>"]]

                predictions.append(" ".join(pred_sentence))
                references.append([" ".join(tgt_sentence)])

    result = bleu.compute(predictions=predictions, references=references)

    return result["bleu"] * 100


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.
    """
    model_config = {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model": model.src_embed.embedding_dim,
        "N": len(model.encoder.layers),
        "num_heads": model.encoder.layers[0].mha.num_heads,
        "d_ff": model.encoder.layers[0].ffn.linear1.out_features,
        "dropout": model.encoder.layers[0].dropout.p,
    }

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    import wandb
    from dataset import Multi30kDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Init W&B
    config = {
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "batch_size": 64,
        "num_epochs": 10,
        "warmup_steps": 4000,
        "lr": 1.0,
    }

    wandb.init(project="da6401-a3", config=config)

    # 2. Build dataset / vocabs
    train_ds = Multi30kDataset("train")
    val_ds   = Multi30kDataset("validation")
    test_ds  = Multi30kDataset("test")

    train_ds.build_vocab()
    val_ds.src_vocab = train_ds.src_vocab
    val_ds.tgt_vocab = train_ds.tgt_vocab
    test_ds.src_vocab = train_ds.src_vocab
    test_ds.tgt_vocab = train_ds.tgt_vocab

    train_data = train_ds.process_data()
    val_data   = val_ds.process_data()
    test_data  = test_ds.process_data()

    pad_idx = train_ds.src_vocab["<pad>"]

    # simple collate_fn
    from torch.nn.utils.rnn import pad_sequence
    def collate_fn(batch):
        src, tgt = zip(*batch)
        src = [torch.tensor(x) for x in src]
        tgt = [torch.tensor(x) for x in tgt]
        src = pad_sequence(src, batch_first=True, padding_value=pad_idx)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=pad_idx)
        return src, tgt

    # 3. DataLoaders
    train_loader = DataLoader(train_data, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_data, batch_size=config["batch_size"],
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_data, batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    # 4. Model
    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    # 5. Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # 6. Scheduler
    from lr_scheduler import NoamScheduler
    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"]
    )

    # 7. Loss
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_ds.tgt_vocab),
        pad_idx=pad_idx,
        smoothing=0.1
    )

    # 8. Training loop
    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch, is_train=True, device=device
        )

        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None,
            epoch, is_train=False, device=device
        )

        save_checkpoint(model, optimizer, scheduler, epoch)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    # 9. BLEU evaluation
    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)

    wandb.log({"test_bleu": bleu})

    print(f"Final BLEU: {bleu:.2f}")
