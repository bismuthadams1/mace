###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional

import torch
import torch.distributed as dist
import logging

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch


# ------------------------------------------------------------------------------
# Helper function for loss reduction that handles DDP correction
# ------------------------------------------------------------------------------
def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        n_local = raw_loss.numel()
        loss_sum = raw_loss.sum()
        total_samples = torch.tensor(
            n_local, device=raw_loss.device, dtype=raw_loss.dtype
        )
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return loss_sum * world_size / total_samples
    return raw_loss.mean()


# ------------------------------------------------------------------------------
# Energy Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.square(ref["energy"] - pred["energy"])
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Calculate per-graph number of atoms.
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_absolute_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)

# -------------------------------------------------------------------------------
# Binary Coupling Loss Function
# -------------------------------------------------------------------------------

# def weighted_classifier_loss(
#     ref: Batch,
#     pred: TensorDict,
#     pos_weight: torch.Tensor,
#     ddp: Optional[bool] = None,
#     global_scale: float = 50.0,   # try 10, 50, 100 to see grads wake up
# ) -> torch.Tensor:
#     #logging info
#     logging.info(f"classifier in {ref["coupling_class"]}")
#     logging.info(f"classifier out {pred["coupling_class"]}")
#     logging.info(f"classifier out (logits) {torch.sigmoid(pred["coupling_class"])}")
#     logging.info(f"classifier in {ref['coupling_class'].shape} shape")
#     logging.info(f"classifier out {pred['coupling_class'].shape} shape")

#     logits = pred["coupling_class"]#.squeeze(-1) TEMPORARILY REMOVE
#     target = ref["coupling_class"].to(logits.dtype)
#     target = target.reshape_as(logits) 
#     logging.info(f"target input with logits {target}")
#     pw = pos_weight.to(device=logits.device, dtype=logits.dtype)


#     per_graph_loss = torch.nn.functional.binary_cross_entropy_with_logits(
#         logits,  # probabilities in logits
#         target,     # 0.0 or 1.0
#         reduction="none",
#         pos_weight = pw
#     )
#     # if you *also* have per-sample weights, include them; else set w=1
#     w = torch.ones_like(per_graph_loss)
#     if hasattr(ref, "weight"):        w = w * ref.weight.reshape(-1).to(logits.dtype, logits.device)
#     if hasattr(ref, "energy_weight"): w = w * ref.energy_weight.reshape(-1).to(logits.dtype, logits.device)

#     loss = (per_graph_loss * w).sum() / w.sum().clamp_min(1e-8) 

#     return reduce_loss(loss, ddp)

def weighted_classifier_loss(
    ref,                   # Batch
    pred,                  # TensorDict
    pos_weight: torch.Tensor,
    ddp: bool | None = None,
    use_balanced_sampler: bool = True,  
):
    logits = pred["coupling_class"].reshape(-1)  # [B]
    target = ref["coupling_class"].to(device=logits.device, dtype=logits.dtype).reshape(-1)

    # If you balance batches with a sampler, don't also reweight positives here.
    pw = (torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
          if use_balanced_sampler
          else pos_weight.to(device=logits.device, dtype=logits.dtype))

    per_ex = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pw
    )  # [B]

    # Per-sample weights (optional). Make sure device/dtype are set via keywords.
    w = torch.ones_like(per_ex)
    if hasattr(ref, "weight"):
        w = w * ref.weight.to(device=logits.device, dtype=logits.dtype).reshape_as(per_ex)
    if hasattr(ref, "energy_weight"):
        w = w * ref.energy_weight.to(device=logits.device, dtype=logits.dtype).reshape_as(per_ex)

    # Proper weighted mean: normalize by sum of weights (not batch size)
    loss = (per_ex * w).sum() / w.sum().clamp_min(1e-8)

    return loss

def gated_regression_l1(ref, pred, ddp=None, beta_scale=1.0):
    # ŷ, y: [B] or [B,1] — make shapes match
    yhat = pred["effective_coupling"]
    y    = ref.effective_coupling.to(yhat.device, yhat.dtype).reshape_as(yhat)

    # mask positives (labels must be 0/1 floats or bools)
    pos = ref.coupling_class.to(dtype=torch.bool).reshape_as(yhat)
    beta_scale = beta_scale
    z_pred = beta_scale * torch.nn.functional.softplus(yhat)  # ensure positivity of predictions
    if pos.any():
        loss = torch.nn.functional.l1_loss(z_pred[pos], y[pos])
    else:
        # keep graph alive even if no positives in this batch
        loss = (z_pred * 0).sum()

    return reduce_loss(loss, ddp)

def gated_regression_huber_log(ref, pred, ddp=None, delta=1.0):
    z = pred["effective_coupling"]        # unconstrained
    y = ref.effective_coupling.to(z.device, z.dtype).reshape_as(z)
    t = torch.log1p(y)
    pos = ref.coupling_class.to(torch.bool).reshape_as(z)

    if pos.any():
        loss = torch.nn.functional.huber_loss(z[pos], t[pos], delta=delta, reduction="mean")
    else:
        loss = (z * 0.0).sum()

    return reduce_loss(loss, ddp)

def loss_regressor_log_hard_gated(ref, pred, beta, mu, sigma, delta=0.5, ddp=None,):
    # model outputs standardized ẑ; targets are y (linear)
    zhat_all = pred["effective_coupling"]
    y        = ref.effective_coupling.to(zhat_all).reshape_as(zhat_all)

    pos = ref.coupling_class.to(torch.bool).reshape_as(zhat_all)
    if not pos.any():
        # keep graph alive but contribute ~0
        return (zhat_all * 0).sum()

    # standardize log1p targets
    z = (torch.log1p(y[pos]/ beta) - mu) / sigma
    zhat = zhat_all[pos]

    # robust loss in z-space
    per = torch.nn.functional.huber_loss(zhat, z, delta=delta, reduction="none")  # [P]

    B = zhat_all.numel()
    # normalize by batch size (NOT by number of positives)
    loss = per.sum() / max(B, 1)

    return reduce_loss(loss, ddp)

def loss_regressor_log_soft_gated(ref, pred, beta, mu, sigma, delta=0.5, ddp=None,):
    # model outputs standardized ẑ; targets are y (linear)
    zhat = pred["effective_coupling"]
    y        = ref.effective_coupling.to(zhat).reshape_as(zhat)
    pos = ref.coupling_class.to(torch.bool).reshape_as(zhat)
    logging.info(f'positives in {pos}')
    ylog = (torch.log1p(y / zhat.new_tensor(beta))
            - zhat.new_tensor(mu)) / zhat.new_tensor(sigma).clamp_min(1e-6)
    alpha = 0.01
    w = ref.coupling_class.to(zhat.dtype).reshape_as(zhat)
    w = alpha + (1.0 - alpha) * w
    logging.info(f'weights going in {w}')

    per = torch.nn.functional.huber_loss(zhat, ylog, delta=delta, reduction="none")
    loss = (w * per).sum() / zhat.numel()   # normalize by B
    logging.info(f'loss in soft gated regressor {loss}, with {pos.count_nonzero().item()} positives')
    return reduce_loss(loss, ddp)


def loss_regressor_log_hard(ref, pred, beta, mu, sigma, delta=0.5):
    # model outputs standardized ẑ; targets are y (linear)
    zhat_all = pred["effective_coupling"]
    y        = ref.effective_coupling.to(zhat_all).reshape_as(zhat_all)

    # standardize log1p targets
    z = (torch.log1p(y/ beta) - mu) / sigma
    zhat = zhat_all

    # robust loss in z-space
    # loss = torch.nn.functional.huber_loss(zhat, z, delta=delta, reduction="mean")

    # try weighted loss
    beta =  torch.tensor(beta, dtype=zhat.dtype, device=zhat.device)
    z    = torch.log1p(y / beta)

    # magnitude-aware weights (alpha ~ 0.5–1.0 is a decent start)
    alpha = 0.75
    w_raw = ((y / (beta + 1e-8)) ** alpha).clamp(0.25, 4.0)
    w     = w_raw / (w_raw.mean().clamp_min(1e-8))

    loss  = torch.nn.functional.huber_loss(zhat, z, delta=delta, reduction="none")
    loss  = (w * loss).mean()

    return loss
# ------------------------------------------------------------------------------
# Graph-level Loss Functions
# ------------------------------------------------------------------------------

def weighted_squared_graph_level_loss(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square((ref["effective_coupling"] - pred["effective_coupling"]))
    )

    logging.info(f"raw loss from graph level loss: {raw_loss}")

    return reduce_loss(raw_loss, ddp)

def weighted_graph_absolute_loss(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    
    logging.info(f'effective coupling in {ref["effective_coupling"]}')
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.abs((ref["effective_coupling"] - pred["effective_coupling"]))
    )

    logging.info(f"raw loss from graph level loss: {raw_loss}")

    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Stress and Virials Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_stress(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_stress_weight
        * torch.square(ref["stress"] - pred["stress"])
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_virials(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Forces Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    return reduce_loss(raw_loss, ddp)


def mean_normed_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.linalg.vector_norm(ref["forces"] - pred["forces"], ord=2, dim=-1)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Dipole Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_dipole(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    raw_loss = torch.square((ref["dipole"] - pred["dipole"]) / num_atoms)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Conditional Losses for Forces
# ------------------------------------------------------------------------------


def conditional_mse_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    # Define multiplication factors for different regimes.
    factors = torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref["forces"].device, dtype=ref["forces"].dtype
    )
    err = ref["forces"] - pred["forces"]
    se = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]
    raw_loss = configs_weight * configs_forces_weight * se
    return reduce_loss(raw_loss, ddp)


def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref_forces.device, dtype=ref_forces.dtype
    )
    norm_forces = torch.norm(ref_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(
        ref_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return reduce_loss(se, ddp)


# ------------------------------------------------------------------------------
# Loss Modules Combining Multiple Quantities
# ------------------------------------------------------------------------------


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.forces_weight * loss_forces

    def __repr__(self):
        return f"{self.__class__.__name__}(forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_stress = weighted_mean_squared_stress(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        # We store the huber_delta rather than a loss with fixed reduction.
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="none", delta=self.huber_delta
            )
            loss_forces = reduce_loss(loss_forces, ddp)
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="none", delta=self.huber_delta
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="mean", delta=self.huber_delta
            )
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="mean", delta=self.huber_delta
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
        configs_energy_weight = ref.energy_weight
        configs_forces_weight = torch.repeat_interleave(
            ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="none",
                delta=self.huber_delta,
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="mean",
                delta=self.huber_delta,
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_virials = weighted_mean_squared_virials(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.virials_weight * loss_virials
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = (
            weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        )  # scale adjustment
        return self.dipole_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f})"


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )


class WeightedEnergyForcesL1L2Loss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_absolute_error_energy(ref, pred, ddp)
        loss_forces = mean_normed_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )
    
class ClassifierLoss(torch.nn.Module):
    def __init__(self, pos_weight, energy_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "pos_weight", 
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
        )


    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = weighted_classifier_loss(ref, pred, self.pos_weight, ddp)
        return self.energy_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(coupling_weight={self.energy_weight:.3f})"

class EffectiveCouplingLoss(torch.nn.Module):
    def __init__(self,
                pos_weight,
                energy_weight=1.0,
                beta_scale: float = 1.0,

        ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "pos_weight", 
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer("mu",   torch.tensor(0.0)) 
        self.register_buffer("sigma",torch.tensor(1.0))
        self.register_buffer("beta",  torch.tensor(beta_scale, dtype=torch.get_default_dtype()))

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        # loss = weighted_squared_graph_level_loss(ref, pred, ddp)
        loss_regressor = loss_regressor_log_hard(
            ref = ref,
            pred =pred,
            beta = self.beta,
            mu = self.mu,
            sigma= self.sigma
        )        
        return self.energy_weight * loss_regressor

    def update_log_space(self, beta=None, mu=None, sigma=None):
        if beta is not None:  self.beta.fill_(float(beta))
        if mu   is not None:  self.mu.fill_(float(mu))
        if sigma is not None: self.sigma.fill_(float(sigma))

    def set_pos_weight(self, pos_weight: float):
        self.pos_weight_buf.fill_(float(pos_weight))

    def to_linear_space(self, t_hat):
        return self.beta * (torch.expm1(self.sigma * t_hat + self.mu))

    def __repr__(self):
        return (f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f})"
        f"beta={self.beta.item():.3f}, mu={self.mu.item():.3f}, sigma={self.sigma.item():.3f}")
    
# loss.py (or wherever you define your combined loss)
class GatedEffectiveCouplingLoss(torch.nn.Module):
    def __init__(self,
                pos_weight,
                beta_scale: float = 1.0,
                energy_weight: float = 1.0,
                classifier_weight: float = 1.0,
                neg_z_penalty: float = 1e-3,
                ):
        super().__init__()
        self.register_buffer( #parameters to be loaded into the state dict
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "pos_weight", 
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer("beta",  torch.tensor(beta_scale, dtype=torch.get_default_dtype()))
        self.register_buffer("pos_weight_buf", torch.tensor(float(pos_weight), dtype=torch.get_default_dtype()))
        # self.register_buffer("beta", torch.tensor(1.0))
        self.register_buffer("mu",   torch.tensor(0.0)) 
        self.register_buffer("sigma",torch.tensor(1.0))
        # self.beta_scale = torch.tensor(beta_scale, dtype=torch.get_default_dtype())
        self.energy_weight = torch.tensor(energy_weight, dtype=torch.get_default_dtype())
        self.classifier_weight = torch.tensor(classifier_weight, dtype=torch.get_default_dtype())
        self.neg_z_penalty = neg_z_penalty
 

    def forward(self, ref, pred, ddp=None) -> torch.Tensor:
        # --- classifier ---
        loss_classifier = weighted_classifier_loss(ref= ref, pred = pred, ddp=ddp, pos_weight=self.pos_weight)
        # --- regression ---
        # loss_regressor = loss_regressor_log_hard_gated(
        #     ref = ref,
        #     pred =pred,
        #     beta = self.beta,
        #     mu = self.mu,
        #     sigma= self.sigma,
        #     ddp=ddp
        #)
        loss_regressor = loss_regressor_log_soft_gated(
            ref = ref,
            pred =pred,
            beta = self.beta,
            mu = self.mu,
            sigma= self.sigma,
            ddp=ddp
        )
        logging.info(f'loss regressor {loss_regressor}')
        logging.info(f'loss classifier {loss_classifier}')

        return self.energy_weight * loss_regressor + self.classifier_weight * loss_classifier
    
    def update_log_space(self, beta=None, mu=None, sigma=None):
        if beta is not None:  self.beta.fill_(float(beta))
        if mu   is not None:  self.mu.fill_(float(mu))
        if sigma is not None: self.sigma.fill_(float(sigma))

    def set_pos_weight(self, pos_weight: float):
        self.pos_weight_buf.fill_(float(pos_weight))

    def to_linear_space(self, t_hat):
        return self.beta * (torch.expm1(self.sigma * t_hat + self.mu))

    def __repr__(self):  #printable representation of an object
        return (f"{self.__class__.__name__}("
                f"pos_weight={self.pos_weight.item():.3f}, "
                f"beta={self.beta.item():.3f}, mu={self.mu.item():.3f}, sigma={self.sigma.item():.3f}, "
                f"coupling_weight={self.energy_weight:.3f}, "
                f"classifier_weight={self.classifier_weight:.3f})")