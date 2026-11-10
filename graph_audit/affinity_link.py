"""Build the affinity model's input on the device from the structure model's output.

CLI path (what this replaces):
  structure model -> writer saves the top-ranked sample to pre_affinity_<id>.npz
  -> load + tokenize -> AffinityCropper (<= 256 tokens, <= 2048 atoms, <= 200
  protein tokens, grown around the ligand) -> Boltz2Featurizer(compute_affinity)
  -> affinity model.

Here:
  structure feats + coordinates of the top-ranked sample -> crop mask (same
  greedy rule as AffinityCropper, as tensor code) -> gather the cropped
  rows/columns of the 34 features the affinity model reads -> affinity model.

Every gathered feature equals what the CLI featurizer produces for the crop, except
`ref_pos`: the featurizer applies a fresh random rotation to each residue's
reference conformer, and here the structure stage's (equally random) rotation is
reused. `crop_and_gather(..., pad_tokens, pad_atoms)` pads to fixed sizes so the
whole thing has static shapes and can be exported.

Batch size 1, like the CLI's affinity step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from boltz.data import const

TOKEN_KEYS = ["affinity_token_mask", "asym_id", "cyclic_period", "deletion_mean", "entity_id",
              "modified", "mol_type", "profile", "res_type", "residue_index", "sym_id",
              "token_pad_mask"]
PAIR_KEYS = ["contact_conditioning", "contact_threshold", "token_bonds", "type_bonds"]
MSA_KEYS = ["msa", "msa_mask", "msa_paired", "deletion_value", "has_deletion"]
ATOM_KEYS = ["atom_pad_mask", "ref_atom_name_chars", "ref_charge", "ref_element", "ref_pos"]
# Also built: token_index, method_feature, ref_space_uid, atom_to_token,
# token_to_rep_atom, frames_idx, profile_affinity, deletion_mean_affinity.


def crop_mask(feats, center_coords, max_tokens=256, max_atoms=2048,
              max_protein_tokens=200, neighborhood_size=10):
    """Tokens AffinityCropper.crop keeps, as an (N,) bool mask.

    center_coords: (N, 3) coordinates of each token's center atom in the
    predicted structure. The CLI walks tokens by distance to the ligand and adds
    each one's neighborhood until a limit would be exceeded. Here every token's
    neighborhood is computed at once, then the step that first exceeds a limit.
    """
    real = feats["token_pad_mask"][0].bool()
    asym = feats["asym_id"][0]
    res = feats["residue_index"][0]
    n = real.shape[0]
    big = torch.iinfo(torch.long).max

    # Distance of each token to the nearest affinity-ligand token.
    lig = real & feats["affinity_token_mask"][0].bool()
    d = ((center_coords[:, None, :] - center_coords[None, :, :]) ** 2).sum(-1) ** 0.5
    dist = torch.where(lig[None, :], d, torch.inf).min(dim=-1).values
    order = torch.argsort(torch.where(real, dist, torch.inf), stable=True)
    rank = torch.empty_like(order).scatter_(0, order, torch.arange(n, device=order.device))

    # Neighborhood of each query token i: its whole chain if the chain has at
    # most `neighborhood_size` tokens, else the smallest residue window
    # [res_i - k, res_i + k] in the chain with at least that many tokens (k <= 10).
    same_chain = (asym[:, None] == asym[None, :]) & real[None, :]
    chain_len = same_chain.sum(-1)
    dres = (res[None, :] - res[:, None]).abs()
    counts = torch.stack([(same_chain & (dres <= k)).sum(-1)
                          for k in range(neighborhood_size + 1)])
    enough = counts >= neighborhood_size
    k_star = torch.where(enough.any(0), enough.int().argmax(0), neighborhood_size)
    nb = same_chain & ((chain_len[:, None] <= neighborhood_size) | (dres <= k_star[:, None]))

    # Step at which each token first enters the crop, then running totals.
    first = torch.where(nb & real[:, None], rank[:, None], big).min(dim=0).values
    steps = torch.arange(n, device=first.device)
    covered = first[None, :] <= steps[:, None]  # (step, token)
    atom_num = (feats["atom_to_token"][0] * feats["atom_pad_mask"][0][:, None]).sum(0).long()
    ligand_ids = real & (feats["mol_type"][0] == const.chain_type_ids["NONPOLYMER"])
    n_tokens = covered.sum(-1)
    n_atoms = (covered * atom_num[None, :]).sum(-1)
    n_protein = (covered & ~ligand_ids[None, :]).sum(-1)
    over = (steps < real.sum()) & (
        (n_tokens > max_tokens) | (n_atoms > max_atoms) | (n_protein > max_protein_tokens)
    )
    stop = torch.where(over.any(), over.int().argmax(), real.sum())
    return first < stop


def _index_and_valid(mask, size):
    """First `size` positions of mask (in order), padded; and which are real."""
    order = torch.argsort((~mask).to(torch.int8), stable=True)[:size]
    valid = torch.arange(order.shape[0], device=mask.device) < mask.sum()
    return order, valid


def _pad_to(x, dim, size, value=0):
    extra = size - x.shape[dim]
    if extra <= 0:
        return x
    pad = [0, 0] * (x.dim() - dim - 1) + [0, extra]
    return F.pad(x, pad, value=value)


def crop_and_gather(feats, token_keep, pad_tokens=None, pad_atoms=None):
    """Affinity-model features for the cropped tokens.

    pad_tokens / pad_atoms = None: sizes follow the crop (atoms padded to a
    multiple of 32, as the featurizer does); this reads the crop size on the
    host. Ints: fixed sizes with padding (static shapes, exportable).
    """
    n = token_keep.shape[0]
    atom_token = feats["atom_to_token"][0].argmax(-1)
    atom_keep = token_keep[atom_token] & feats["atom_pad_mask"][0].bool()
    if pad_tokens is None:
        n_tok = int(token_keep.sum())
        n_atom = int(atom_keep.sum())
        pad_tokens, pad_atoms = n_tok, -(-n_atom // 32) * 32
    t, t_valid = _index_and_valid(token_keep, min(pad_tokens, n))
    a, a_valid = _index_and_valid(atom_keep, min(pad_atoms, atom_keep.shape[0]))

    def tok(x, dim):  # gather along a token dim, zero the padding
        y = x.index_select(dim, t)
        shape = [1] * y.dim()
        shape[dim] = -1
        return _pad_to(y * t_valid.view(shape).to(y.dtype), dim, pad_tokens)

    def atm(x, dim):
        y = x.index_select(dim, a)
        shape = [1] * y.dim()
        shape[dim] = -1
        return _pad_to(y * a_valid.view(shape).to(y.dtype), dim, pad_atoms)

    out = {}
    for k in TOKEN_KEYS:
        out[k] = tok(feats[k], 1)
    for k in PAIR_KEYS:
        out[k] = tok(tok(feats[k], 1), 2)
    for k in MSA_KEYS:
        out[k] = tok(feats[k], 2)
    # The featurizer pads the MSA's token dim with the gap token.
    gap = const.token_ids["-"]
    tv = _pad_to(t_valid, 0, pad_tokens, value=False)
    out["msa"] = torch.where(tv[None, None, :], out["msa"], gap)
    for k in ATOM_KEYS:
        out[k] = atm(feats[k], 1)
    out["atom_to_token"] = tok(atm(feats["atom_to_token"], 1), 2)
    out["token_to_rep_atom"] = atm(tok(feats["token_to_rep_atom"], 1), 2)

    # Renumbered after the crop, as the featurizer does.
    out["token_index"] = torch.arange(pad_tokens, device=t.device)[None] * tv[None]
    out["method_feature"] = torch.full_like(out["mol_type"], const.method_types_ids["other"]) * tv[None]
    # ref_space_uid: index of each (chain, residue) in order of first appearance.
    uid = feats["ref_space_uid"][0].index_select(0, a)
    new_res = torch.cat([torch.zeros_like(uid[:1]), (uid[1:] != uid[:-1]).long()])
    out["ref_space_uid"] = _pad_to((new_res.cumsum(0) * a_valid)[None], 1, pad_atoms)

    # Frame atom indices point into the cropped atom table.
    old_to_new = torch.zeros(atom_keep.shape[0], dtype=torch.long, device=t.device)
    old_to_new = old_to_new.scatter(0, a, torch.arange(a.shape[0], device=t.device))
    frames = old_to_new[feats["frames_idx"][0, 0].index_select(0, t)]
    out["frames_idx"] = _pad_to((frames * t_valid[:, None])[None, None], 2, pad_tokens)

    # Single-sequence profile (MSA query row only), as compute_affinity adds.
    query = feats["msa"][0, 0].index_select(0, t)
    out["profile_affinity"] = _pad_to(
        (F.one_hot(query, const.num_tokens).float() * t_valid[:, None])[None], 1, pad_tokens)
    out["deletion_mean_affinity"] = _pad_to(
        (feats["deletion_value"][0, 0].index_select(0, t) * t_valid)[None], 1, pad_tokens)
    return out


def center_coords(feats, atom_coords):
    """(N, 3) center-atom coordinates of each token from (M, 3) atom coordinates."""
    center_atom = feats["token_to_center_atom"][0].argmax(-1)
    return atom_coords.index_select(0, center_atom)
