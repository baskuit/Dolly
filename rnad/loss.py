import torch

from typing import Sequence
from math import ceil


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


def backpropagate(
    model: torch.nn.Module,
    grad_list_dict,
    batch,
    value_targets,
    policy_targets,
    has_playeds,
    valid: torch.Tensor,
    player_id: torch.Tensor,
    legal_actions: torch.Tensor,
    clip: int = 10_000,
    threshold: float = 2.0,
):
    T, B, *_ = batch["public_dex"].shape
    num_batch_splits = ceil(B / model.max_batch_split)
    num_traj_splits = ceil(T / model.max_traj_split)

    minibatch_max_len = valid.sum(0)
    minibatch_max_len = minibatch_max_len.view(num_batch_splits, B // num_batch_splits)
    minibatch_max_len = minibatch_max_len.max(-1).values.tolist()

    is_vector = torch.unsqueeze(torch.ones_like(valid), dim=-1)
    importance_sampling_corrections = [is_vector] * 2

    total_value = valid.sum()

    neurd_losses = []
    value_losses = []
    total_losses = []

    for b in range(num_batch_splits):
        min_iter = ceil(minibatch_max_len[b] / model.max_traj_split)
        for t in range(min(min_iter, num_traj_splits)):
            minibatch = {
                key: value[
                    model.max_traj_split * t : model.max_traj_split * (t + 1),
                    model.max_batch_split * b : model.max_batch_split * (b + 1),
                ]
                for key, value in batch.items()
            }
            predictions = model(minibatch)

            minibatch_value_target = [
                value_target[
                    model.max_traj_split * t : model.max_traj_split * (t + 1),
                    model.max_batch_split * b : model.max_batch_split * (b + 1),
                ]
                for value_target in value_targets
            ]
            minibatch_has_played = [
                has_played[
                    model.max_traj_split * t : model.max_traj_split * (t + 1),
                    model.max_batch_split * b : model.max_batch_split * (b + 1),
                ]
                for has_played in has_playeds
            ]

            minibatch_policy_target = [
                policy_target[
                    model.max_traj_split * t : model.max_traj_split * (t + 1),
                    model.max_batch_split * b : model.max_batch_split * (b + 1),
                ]
                for policy_target in policy_targets
            ]
            minibatch_valid = valid[
                model.max_traj_split * t : model.max_traj_split * (t + 1),
                model.max_batch_split * b : model.max_batch_split * (b + 1),
            ]
            minibatch_player_id = player_id[
                model.max_traj_split * t : model.max_traj_split * (t + 1),
                model.max_batch_split * b : model.max_batch_split * (b + 1),
            ]
            minibatch_legal_actions = legal_actions[
                model.max_traj_split * t : model.max_traj_split * (t + 1),
                model.max_batch_split * b : model.max_batch_split * (b + 1),
            ]
            minibatch_isc = [
                isc[
                    model.max_traj_split * t : model.max_traj_split * (t + 1),
                    model.max_batch_split * b : model.max_batch_split * (b + 1),
                ]
                for isc in importance_sampling_corrections
            ]

            accumulation_step_weight = minibatch_valid.sum() / total_value
            loss_weight = accumulation_step_weight.item()

            loss_v = (
                get_loss_v(
                    [predictions["value"]] * 2,
                    minibatch_value_target,
                    minibatch_has_played,
                )
                * loss_weight
            )

            loss_nerd = (
                get_loss_nerd(
                    [predictions["logits"]] * 2,
                    [predictions["policy"]] * 2,
                    minibatch_policy_target,
                    minibatch_valid,
                    minibatch_player_id,
                    minibatch_legal_actions,
                    minibatch_isc,
                    clip=clip,
                    threshold=threshold,
                )
                * loss_weight
            )

            loss = loss_v + loss_nerd
            neurd_losses.append(loss_nerd)
            value_losses.append(loss_v)
            total_losses.append(loss)

            loss.backward()

    neurd_loss = sum(neurd_losses).item()
    value_loss = sum(value_losses).item()

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if key not in grad_list_dict:
            grad_list_dict[key] = [value.grad]
        else:
            grad_list_dict[key].append(value.grad)

    return neurd_loss, value_loss
