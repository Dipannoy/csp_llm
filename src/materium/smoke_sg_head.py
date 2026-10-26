"""Does the auxiliary space-group head train without blowing up?

Checks the three things that would waste a multi-day run if wrong:
  - the head exists only when the weight is positive
  - forward+backward produce a finite loss and real gradients into the head
  - masked-out labels (unknown space group) are excluded, not learned
"""
import torch
from materium.coord_loss import CoordLossConfig, TokenLayout
from materium.model import ConditionConfig, LLamaTransformer, ModelArgs, RopeMode

B, S, NBINS, NSPEC, NSPECIAL = 4, 58, 1024, 20, 4
VOCAB = NSPECIAL + NSPEC + NBINS
layout = TokenLayout(atoms_token_id=1, lattice_token_id=2, pad_id=0,
                     quant_offset=NSPECIAL + NSPEC, num_quant_bins=NBINS,
                     block_size=4, coords_first=False,
                     coord_tiebreak_descending=True)

args = ModelArgs(dim=64, n_layers=2, n_heads=4, n_kv_heads=4,
                 vocab_size=VOCAB, max_seq_len=128, dropout=0.0,
                 condition_config={"space_group": ConditionConfig("embedding", 231, 64)})

def build(w):
    return LLamaTransformer(args, token_layout=layout,
                            coord_loss_config=CoordLossConfig(sigma_bins=2.0),
                            sg_aux_weight=w)

print("head absent when weight is 0 :", build(0.0).sg_head is None)
m = build(0.5)
print("head present when weight > 0 :", m.sg_head is not None)

tokens = torch.randint(NSPECIAL, VOCAB, (B, S))
tokens[:, 0], tokens[:, 1] = 3, layout.atoms_token_id
targets = tokens.clone()
roles = torch.full((B, S), 3, dtype=torch.long)

sg = torch.tensor([221, 62, 1, 139])
mask = torch.tensor([False, False, True, False])   # third label is unknown

m.train()
m(tokens, targets, conditions={}, target_roles=roles, sg_targets=sg, sg_mask=mask)
loss, met = m.last_loss, m.last_metrics
print("loss finite                  :", bool(torch.isfinite(loss)))
print("sg_aux_loss                  : %.4f" % met["sg_aux_loss"])
print("sg_aux_acc                   : %.4f" % met["sg_aux_acc"])
print("sg_aux_frac (3 of 4 kept)    : %.4f" % met["sg_aux_frac"])

loss.backward()
g = m.sg_head.weight.grad
print("gradient reaches sg_head     :", g is not None and bool(g.abs().sum() > 0))

# all labels unknown -> term must be skipped, not NaN
m.zero_grad()
m(tokens, targets, conditions={}, target_roles=roles,
  sg_targets=sg, sg_mask=torch.ones(B, dtype=torch.bool))
print("all-masked -> term skipped   :", "sg_aux_loss" not in m.last_metrics)
print("all-masked loss still finite :", bool(torch.isfinite(m.last_loss)))

# weight 0 path must be untouched
m0 = build(0.0)
m0(tokens, targets, conditions={}, target_roles=roles, sg_targets=sg, sg_mask=mask)
print("weight=0 adds no sg metrics  :", "sg_aux_loss" not in m0.last_metrics)
print()
print("SMOKE TEST DONE")
