import torch

from typing import Sequence


def get_loss_v(
    v_list: Sequence[torch.Tensor],
    v_target_list: Sequence[torch.Tensor],
    mask_list: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Define the loss function for the critic."""
    loss_v_list = []
    for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
        assert v_n.shape[0] == v_target.shape[0]

        loss_v = torch.unsqueeze(mask, dim=-1) * (v_n - v_target.detach()) ** 2
        normalization = torch.sum(mask)
        loss_v = torch.sum(loss_v) / (normalization + (normalization == 0.0))

        loss_v_list.append(loss_v)

    return sum(loss_v_list)


def get_loss_nerd(
    logit_list: Sequence[torch.Tensor],
    policy_list: Sequence[torch.Tensor],
    q_vr_list: Sequence[torch.Tensor],
    valid: torch.Tensor,
    player_ids: Sequence[torch.Tensor],
    legal_actions: torch.Tensor,
    importance_sampling_correction: Sequence[torch.Tensor],
    clip: float = 100,
    threshold: float = 2,
) -> torch.Tensor:
    """Define the nerd loss."""
    assert isinstance(importance_sampling_correction, list)
    loss_pi_list = []
    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
        zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
    ):
        assert logit_pi.shape[0] == q_vr.shape[0]
        # loss policy
        adv_pi = q_vr - torch.sum(pi * q_vr, dim=-1, keepdim=True)
        adv_pi = is_c * adv_pi  # importance sampling correction
        adv_pi = torch.clip(adv_pi, min=-clip, max=clip)
        adv_pi = adv_pi.detach()

        logits = logit_pi - torch.mean(logit_pi * legal_actions, dim=-1, keepdim=True)

        threshold_center = torch.zeros_like(logits)

        force_threshold = apply_force_with_threshold(
            logits,
            adv_pi,
            threshold,
            threshold_center,
        )
        nerd_loss = torch.sum(legal_actions * force_threshold, axis=-1)
        nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
        loss_pi_list.append(nerd_loss)

    return sum(loss_pi_list)


def apply_force_with_threshold(
    decision_outputs: torch.Tensor,
    force: torch.Tensor,
    threshold: float,
    threshold_center: torch.Tensor,
) -> torch.Tensor:
    """Apply the force with below a given threshold."""
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    force_negative = torch.clamp(force, max=0.0)
    force_positive = torch.clamp(force, min=0.0)
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * clipped_force.detach()


def renormalize(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The `normalization` is the number of steps over which loss is computed."""
    loss = torch.sum(loss * mask)
    normalization = torch.sum(mask)
    return loss / normalization.clamp(min=1)

