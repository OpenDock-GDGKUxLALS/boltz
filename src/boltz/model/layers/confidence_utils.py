import torch
from torch import nn

from boltz.data import const
from boltz.model.modules.utils import autocast_device_type


def compute_collinear_mask(v1, v2):
    norm1 = torch.norm(v1, dim=1, keepdim=True)
    norm2 = torch.norm(v2, dim=1, keepdim=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = torch.abs(torch.sum(v1 * v2, dim=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_frame_pred(
    pred_atom_coords,
    frames_idx_true,
    feats,
    multiplicity,
    resolved_mask=None,
    inference=False,
):
    if inference:
        return compute_frame_pred_inference(pred_atom_coords, frames_idx_true, feats, multiplicity)
    with torch.amp.autocast("cuda", enabled=False):
        asym_id_token = feats["asym_id"]
        asym_id_atom = torch.bmm(
            feats["atom_to_token"].float(), asym_id_token.unsqueeze(-1).float()
        ).squeeze(-1)

    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    frames_idx_pred = (
        frames_idx_true.clone()
        .repeat_interleave(multiplicity, 0)
        .reshape(B // multiplicity, multiplicity, -1, 3)
    )

    # Iterate through the batch and modify the frames for nonpolymers
    for i, pred_atom_coord in enumerate(pred_atom_coords):
        token_idx = 0
        atom_idx = 0
        for id in torch.unique(asym_id_token[i]):
            mask_chain_token = (asym_id_token[i] == id) * feats["token_pad_mask"][i]
            mask_chain_atom = (asym_id_atom[i] == id) * feats["atom_pad_mask"][i]
            num_tokens = int(mask_chain_token.sum().item())
            num_atoms = int(mask_chain_atom.sum().item())
            if (
                feats["mol_type"][i, token_idx] != const.chain_type_ids["NONPOLYMER"]
                or num_atoms < 3
            ):
                token_idx += num_tokens
                atom_idx += num_atoms
                continue
            dist_mat = (
                (
                    pred_atom_coord[:, mask_chain_atom.bool()][:, None, :, :]
                    - pred_atom_coord[:, mask_chain_atom.bool()][:, :, None, :]
                )
                ** 2
            ).sum(-1) ** 0.5
            if inference:
                resolved_pair = 1 - (
                    feats["atom_pad_mask"][i][mask_chain_atom.bool()][None, :]
                    * feats["atom_pad_mask"][i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            else:
                if resolved_mask is None:
                    resolved_mask = feats["atom_resolved_mask"]
                resolved_pair = 1 - (
                    resolved_mask[i][mask_chain_atom.bool()][None, :]
                    * resolved_mask[i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            frames = (
                torch.cat(
                    [
                        indices[:, :, 1:2],
                        indices[:, :, 0:1],
                        indices[:, :, 2:3],
                    ],
                    dim=2,
                )
                + atom_idx
            )
            try:
                frames_idx_pred[i, :, token_idx : token_idx + num_atoms, :] = frames
            except Exception as e:
                print(f"Failed to process {feats['pdb_id']} due to {e}")
            token_idx += num_tokens
            atom_idx += num_atoms

    frames_expanded = pred_atom_coords[
        torch.arange(0, B // multiplicity, 1)[:, None, None, None].to(
            frames_idx_pred.device
        ),
        torch.arange(0, multiplicity, 1)[None, :, None, None].to(
            frames_idx_pred.device
        ),
        frames_idx_pred,
    ].reshape(-1, 3, 3)

    # Compute masks for collinearity / overlap
    mask_collinear_pred = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(B // multiplicity, multiplicity, -1)
    return frames_idx_pred, mask_collinear_pred * feats["token_pad_mask"][:, None, :]


def compute_frame_pred_inference(pred_atom_coords, frames_idx_true, feats, multiplicity):
    """compute_frame_pred(inference=True) without the per-chain loop."""
    x = pred_atom_coords.reshape(-1, multiplicity, *pred_atom_coords.shape[1:])
    atom_to_token = feats["atom_to_token"].float()
    same_chain = feats["asym_id"][:, :, None] == feats["asym_id"][:, None, :]
    atoms_per_token = (atom_to_token * feats["atom_pad_mask"][:, :, None]).sum(1)
    nonpolymer = feats["mol_type"] == const.chain_type_ids["NONPOLYMER"]
    chain_atoms = (same_chain * atoms_per_token[:, None, :]).sum(-1)
    ligand = nonpolymer & (chain_atoms >= 3) & feats["token_pad_mask"].bool()
    frames = frames_idx_true.repeat_interleave(multiplicity, 0).reshape(*x.shape[:2], -1, 3)
    if frames.shape[2] >= 3:  # a ligand frame needs 3 atoms, so 3 tokens
        # Ligand tokens have one atom each, so the nearest atoms are searched over tokens.
        token_atom = atom_to_token.argmax(dim=1)[:, None].expand(-1, multiplicity, -1)
        xt = torch.gather(x, 2, token_atom[..., None].expand(-1, -1, -1, 3))
        dist = ((xt[:, :, None] - xt[:, :, :, None]) ** 2).sum(-1) ** 0.5
        in_chain = (same_chain & ligand[:, :, None] & ligand[:, None, :])[:, None]
        nearest = torch.where(in_chain, dist, torch.inf).topk(3, -1, largest=False).indices
        nearest = torch.gather(token_atom[:, :, None].expand_as(dist), 3, nearest)
        frames = torch.where(ligand[:, None, :, None], nearest[..., [1, 0, 2]], frames)
    f = torch.gather(x, 2, frames.flatten(2)[..., None].expand(-1, -1, -1, 3)).reshape(-1, 3, 3)
    mask = compute_collinear_mask(f[:, 1] - f[:, 0], f[:, 1] - f[:, 2])
    return frames, mask.reshape(*x.shape[:2], -1) * feats["token_pad_mask"][:, None]


def compute_aggregated_metric(logits, end=1.0):
    # Compute aggregated metric from logits
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    )
    probs = nn.functional.softmax(logits, dim=-1)
    plddt = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


def tm_function(d, Nres):
    d0 = 1.24 * (torch.clip(Nres, min=19) - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (d / d0) ** 2)


def compute_ptms(logits, x_preds, feats, multiplicity):
    # It needs to take as input the mask of the frames as they are not used to compute the PTM
    _, mask_collinear_pred = compute_frame_pred(
        x_preds, feats["frames_idx"], feats, multiplicity, inference=True
    )
    # mask overlapping, collinear tokens and ions (invalid frames)
    mask_pad = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
    maski = mask_collinear_pred.reshape(-1, mask_collinear_pred.shape[-1])
    pair_mask_ptm = maski[:, :, None] * mask_pad[:, None, :] * mask_pad[:, :, None]
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    pair_mask_iptm = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
    )
    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    end = 32.0
    pae_value = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    ).unsqueeze(0)
    N_res = mask_pad.sum(dim=-1, keepdim=True)
    tm_value = tm_function(pae_value, N_res).unsqueeze(1).unsqueeze(2)
    probs = nn.functional.softmax(logits, dim=-1)
    tm_expected_value = torch.sum(
        probs * tm_value,
        dim=-1,
    )  # shape (B, N, N)
    ptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_ptm, dim=-1)
        / (torch.sum(pair_mask_ptm, dim=-1) + 1e-5),
        dim=1,
    ).values
    iptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_iptm, dim=-1)
        / (torch.sum(pair_mask_iptm, dim=-1) + 1e-5),
        dim=1,
    ).values

    # compute ligand and protein iPTM
    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)
    is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    is_protein_token = (token_type == const.chain_type_ids["PROTEIN"]).float()

    ligand_iptm_mask = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (
            (is_ligand_token[:, :, None] * is_protein_token[:, None, :])
            + (is_protein_token[:, :, None] * is_ligand_token[:, None, :])
        )
    )
    protein_ipmt_mask = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (is_protein_token[:, :, None] * is_protein_token[:, None, :])
    )

    ligand_iptm = torch.max(
        torch.sum(tm_expected_value * ligand_iptm_mask, dim=-1)
        / (torch.sum(ligand_iptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values
    protein_iptm = torch.max(
        torch.sum(tm_expected_value * protein_ipmt_mask, dim=-1)
        / (torch.sum(protein_ipmt_mask, dim=-1) + 1e-5),
        dim=1,
    ).values

    # Chain-pair ipTM as (B, N, N) over chains in ascending asym id order: [b, c1, c2] is the
    # old dict's [c1][c2], the best row in chain c2 of the mean TM over columns in chain c1.
    first = ~(asym_id[:, :, None] == asym_id[:, None, :]).tril(-1).any(-1)  # first token per chain
    chain = ((asym_id[:, None, :] < asym_id[:, :, None]) & first[:, None, :]).sum(-1)
    columns = nn.functional.one_hot(chain, asym_id.shape[1]) * mask_pad[..., None]
    with torch.amp.autocast(autocast_device_type(asym_id.device.type), enabled=False):  # fp32 bmm
        row_sums = torch.bmm(tm_expected_value, columns)
    rows = (maski * mask_pad)[..., None] * row_sums / (columns.sum(1, keepdim=True) + 1e-5)
    index = chain[..., None].expand_as(rows)  # rows are >= 0, so a max from 0
    chain_pair_iptm = torch.zeros_like(rows).scatter_reduce(1, index, rows, "amax").transpose(1, 2)
    return ptm, iptm, ligand_iptm, protein_iptm, chain_pair_iptm
