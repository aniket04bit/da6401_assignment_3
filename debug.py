import torch
from model import Transformer, make_src_mask, make_tgt_mask

# small fake vocab
vocab_size = 100

# create model
model = Transformer(vocab_size, vocab_size)

# fake batch
batch_size = 2
src_len = 10
tgt_len = 10

src = torch.randint(0, vocab_size, (batch_size, src_len))
tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

# target shifting (important)
tgt_input = tgt[:, :-1]

# masks
src_mask = make_src_mask(src)
tgt_mask = make_tgt_mask(tgt_input)

# forward pass
out = model(src, tgt_input, src_mask, tgt_mask)

print("Output shape:", out.shape)