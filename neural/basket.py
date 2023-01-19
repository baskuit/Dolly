import torch.nn as nn
from _torch.nn.activation import MultiheadAttention
import neural.attention
import neural.dot_scaled_linear

from poke_env.data import (
    POKEDEX_STOI,
    MOVES_STOI,
    ITEMS_STOI,
    ABILITIES_STOI,
    MOVE_CUTOFF,
    POKEDEX_CUTOFF,
)

from typing import List, NamedTuple
import math
import torch
import logging
from math import ceil

from typing import List, Dict, NamedTuple, Optional
from rnad.loss import get_loss_nerd, get_loss_v

from mp.player import ObservationTensor

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POLICY_SIZE = 9
output_buffer_specs = lambda T, B, device: {
    "policy": {"size": (T, B, POLICY_SIZE), "dtype": torch.float32, "device": device},
    "value": {"size": (T, B, 1), "dtype": torch.float32, "device": device},
    "log_policy": {
        "size": (T, B, POLICY_SIZE),
        "dtype": torch.float32,
        "device": device,
    },
    "logits": {"size": (T, B, POLICY_SIZE), "dtype": torch.float32, "device": device},
}


class Inference(NamedTuple):
    pi: torch.Tensor
    v: torch.Tensor
    log_pi: torch.Tensor
    logit: torch.Tensor


class EncodedObservation(NamedTuple):
    words: list
    moves: torch.Tensor


def _legal_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """A soft-max policy that respects legal_actions."""
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    l_min = logits.min(axis=-1, keepdim=True).values
    logits = torch.where(legal_actions, logits, l_min)
    logits -= logits.max(axis=-1, keepdim=True).values
    logits *= legal_actions
    exp_logits = torch.where(
        legal_actions, torch.exp(logits), 0
    )  # Illegal actions become 0.
    exp_logits_sum = torch.sum(exp_logits, axis=-1, keepdim=True)
    policy = torch.where(legal_actions, exp_logits / exp_logits_sum, 0)
    return policy


def legal_log_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """Return the log of the policy on legal action, 0 on illegal action."""
    # logits_masked has illegal actions set to -inf.
    logits_masked = logits + torch.log(legal_actions)
    max_legal_logit = logits_masked.max(axis=-1, keepdim=True).values
    logits_masked = logits_masked - max_legal_logit
    # exp_logits_masked is 0 for illegal actions.
    exp_logits_masked = torch.exp(logits_masked)

    baseline = torch.log(torch.sum(exp_logits_masked, axis=-1, keepdim=True))
    # Subtract baseline from logits. We do not simply return
    #     logits_masked - baseline
    # because that has -inf for illegal actions, or
    #     legal_actions * (logits_masked - baseline)
    # because that leads to 0 * -inf == nan for illegal actions.
    log_policy = torch.multiply(legal_actions, (logits - max_legal_logit - baseline))
    return log_policy


class ObservationEncoder(nn.Module):

    """
    Pre-processing before trunk
    """

    def __init__(
        self,
        move_features=2**4,
        our_active_features=2**7,
        our_bench_features=2**6,
        opp_active_features=2**7,
        opp_bench_features=2**6,
        dex_=8,
        m_id=8,
        m_pp=4,
        moveslot_=16,
        stat_=16,
        hp_=8,
        status_=4,
        weather_=8,
        slot_=16,
        item_=4,
        ability_=8,
        boosts_vol_=48,
        f_public=64,
        f_private=80,
        f_active=32,
        f_field=16,
    ):
        super().__init__()

        self.dex_emb = nn.Embedding(len(POKEDEX_STOI) + 1, dex_)
        self.move_id_emb = nn.Embedding(len(MOVES_STOI) + 1, m_id)
        self.move_pp_emb = nn.Embedding(65, m_pp)
        self.move_linear = nn.Linear(m_id + m_pp, move_features, bias=True)
        self.moveset_linear = nn.Linear(move_features, moveslot_, bias=False)
        self.item_emb = nn.Embedding(len(ITEMS_STOI) + 1, item_)
        self.ability_emb = nn.Embedding(len(ABILITIES_STOI) + 1, ability_)
        self.hp_linear = nn.Linear(1, hp_, bias=False)
        self.status_emb = nn.Embedding(8, status_)
        self.stat_linear = nn.Linear(6, stat_, bias=False)
        # self.dmia_linear = nn.Linear(
        #     dex_ + moveslot_ + item_ + ability_, dmia_, bias=False
        # )
        self.weather_linear = nn.Linear(8, weather_)
        self.slot_linear = nn.Linear(20, slot_, bias=False)
        self.boosts_vol_linear = nn.Linear(171, boosts_vol_, bias=False)

        self.public_linear = nn.Linear(
            dex_ + moveslot_ + item_ + ability_ + hp_ + status_, f_public, bias=False
        )
        self.private_linear = nn.Linear(
            dex_ + moveslot_ + item_ + ability_ + stat_, f_private, bias=False
        )
        self.active_linear = nn.Linear(boosts_vol_, f_active, bias=False)
        self.field_linear = nn.Linear(weather_ + slot_, f_field, bias=False)
        self.our_active = nn.Linear(
            f_public + f_private + f_active + f_field, our_active_features
        )
        self.our_bench = nn.Linear(f_public + f_private + f_field, our_bench_features)
        self.opp_active = nn.Linear(f_public + f_active + f_field, opp_active_features)
        self.opp_bench = nn.Linear(f_public + f_field, opp_bench_features)

    def forward(self, obs_dict: ObservationTensor):

        pub_priv_dex_emb = self.dex_emb(
            torch.cat((obs_dict["public_dex"], obs_dict["private_dex"]), dim=-2)[..., 0]
        )  # (..., 18, d_=8)
        all_moves = torch.cat(
            (
                obs_dict["public_moves"],
                obs_dict["private_moves"],
                obs_dict["active_moves"],
            ),
            dim=-3,
        )  # (..., 19, 4, 2)
        all_moves_id_emb = self.move_id_emb(all_moves[..., 0])  # (..., 19, 4, 8)
        all_moves_pp_emb = self.move_pp_emb(all_moves[..., 1])  # (..., 19, 4, 8)
        all_moves_linear = torch.relu(
            self.move_linear(torch.cat((all_moves_id_emb, all_moves_pp_emb), dim=-1))
        )  # (..., 19, 4, m_)
        all_moveset_linear = torch.sum(
            all_moves_linear[..., :18, :], dim=-2
        )  # (..., 18, m_)
        enc_all_moveset = self.moveset_linear(all_moveset_linear)  # (..., 18, ms_)
        pub_priv_item_emb = self.item_emb(
            torch.cat((obs_dict["public_item"], obs_dict["private_item"]), dim=-2)[
                ..., 0
            ]
        )  # (..., 18, i_=4)
        pub_priv_ability_emb = self.ability_emb(
            torch.cat(
                (
                    obs_dict["public_ability"],
                    obs_dict["private_ability"],
                ),
                dim=-2,
            )[..., 0]
        )  # (..., 18, a_=8)
        dmia = torch.cat(
            (
                pub_priv_dex_emb,
                enc_all_moveset[..., :18, :],
                pub_priv_item_emb,
                pub_priv_ability_emb[..., :18, :],
            ),
            dim=-1,
        )
        emb_weather = self.weather_linear(obs_dict["weather"])
        emb_slots = self.slot_linear(obs_dict["slot"])  # (..., 2, 16)

        # # # #

        enc_public = self.public_linear(
            torch.cat(
                (
                    dmia[..., :12, :],
                    self.hp_linear(obs_dict["public_hp"]),
                    self.status_emb(obs_dict["public_status"][..., 0]),
                ),
                dim=-1,
            )
        )  # (..., 12, 64)
        enc_private = self.private_linear(
            torch.cat(
                (self.stat_linear(obs_dict["private_stats"]), dmia[..., 12:, :]), dim=-1
            )
        )  # (..., 6, 96)
        enc_field = self.field_linear(
            torch.cat(
                (
                    torch.cat((emb_weather, emb_slots[..., :1, :]), dim=-1),
                    torch.cat((emb_weather, emb_slots[..., 1:, :]), dim=-1),
                ),
                dim=-2,
            )
        )  # (..., 2, 32)
        enc_field = torch.repeat_interleave(
            enc_field, repeats=6, dim=-2
        )  # (..., 12, 32)
        enc_active = self.active_linear(
            torch.cat((self.boosts_vol_linear(obs_dict["active_boosts_vol"]),), dim=-1)
        )  # (..., 2, 64)

        # # # #
        our_active = self.our_active(
            torch.cat(
                (
                    enc_public[..., :1, :],
                    enc_private[..., :1, :],
                    enc_active[..., :1, :],
                    enc_field[..., :1, :],
                ),
                dim=-1,
            )
        )
        our_bench = self.our_bench(
            torch.cat(
                (
                    enc_public[..., 1:6, :],
                    enc_private[..., 1:6, :],
                    enc_field[..., 1:6, :],
                ),
                dim=-1,
            )
        )
        opp_active = self.opp_active(
            torch.cat(
                (
                    enc_public[..., 6:7, :],
                    enc_active[..., 1:2, :],
                    enc_field[..., 6:7, :],
                ),
                dim=-1,
            )
        )
        opp_bench = self.opp_bench(
            torch.cat((enc_public[..., 7:, :], enc_field[..., 7:, :]), dim=-1)
        )

        words = [our_active, our_bench, opp_active, opp_bench]
        words = [torch.nn.functional.normalize(_, dim=-1, p=2) for _ in words]
        moves = torch.nn.functional.normalize(
            all_moves_linear[..., 18, :, :], dim=-1, p=2
        )
        return EncodedObservation(
            words=words,
            moves=moves,  # (... 4, m_)
        )


class PolicyHead(torch.nn.Module):
    def __init__(
        self,
        group_features=(2**7, 2**6, 2**7, 2**6),
        depth_1=1,
        depth_2=1,
        dot_dim_scale_1=0.5,
        dot_dim_scale_2=0.5,
        tower_dim=2**8,
        logit_dim=2**9,
        move_features=32,
        bench_features=2**8,
    ) -> None:

        super().__init__()

        self.move_linear = torch.nn.Linear(move_features, tower_dim)
        self.switch_linear = torch.nn.Linear(bench_features, tower_dim)
        self.logit_linear_1 = torch.nn.Linear(tower_dim, logit_dim)
        self.logit_linear_2 = torch.nn.Linear(logit_dim, 1)

        self.tower1 = neural.dot_scaled_linear.CrossDSLTower(
            depth=depth_1,
            row_dims=(tower_dim, tower_dim),
            col_dims=group_features,
            dot_dim_scale=dot_dim_scale_1,
            group_lengths=(4, 5),
        )
        self.tower2 = neural.dot_scaled_linear.SelfDSLTower(
            depth=depth_2,
            row_dims=(tower_dim, tower_dim),
            dot_dim_scale=dot_dim_scale_2,
            group_lengths=(4, 5),
        )

        self.norm_move = torch.nn.LayerNorm((4, tower_dim))
        self.norm_switch = torch.nn.LayerNorm((5, tower_dim))
        self.norm_tower_out = torch.nn.LayerNorm((9, tower_dim))

    def forward(self, active_moves, latent_words):
        latent_our_bench = latent_words[1]
        tower_in = [
            self.norm_move(self.move_linear(active_moves)),
            self.norm_switch(self.switch_linear(latent_our_bench)),
        ]
        tower_out = self.norm_tower_out(
            torch.cat(self.tower2(self.tower1(tower_in, latent_words)), dim=-2)
        )
        pre_logits = self.logit_linear_1(tower_out)
        logits = self.logit_linear_2(torch.relu(pre_logits))[..., 0]
        return logits


class ValueHead(torch.nn.Module):
    def __init__(
        self,
        depth=2,
        group_features=(2**7, 2**6, 2**7, 2**6),
        dot_dim_scale=0.5,
        value_dim=2**8,
    ) -> None:
        super().__init__()
        self.tower = neural.dot_scaled_linear.SelfDSLTower(
            depth=depth,
            row_dims=group_features,
            dot_dim_scale=dot_dim_scale,
        )
        self.linear_inner = torch.nn.Linear(
            2 * group_features[0] + 10 * group_features[1], value_dim
        )
        self.linear_outer = torch.nn.Linear(value_dim, 1)

    def forward(self, words):
        words = self.tower(words)
        blob = torch.concat([torch.flatten(w, -2, -1) for w in words], dim=-1)
        inner = torch.relu(self.linear_inner(blob))
        value = self.linear_outer(inner)
        return value


class BasketNet(nn.Module):
    def __init__(
        self,
        depth_body=2,
        depth_policy_cross=1,
        depth_policy_self=1,
        depth_value=1,
        group_features=(2**7, 2**6, 2**7, 2**6),
        move_features=16,
        dot_dim_scale_body=1,
        dot_dim_scale_value=1,
        dot_dim_scale_policy=1,
        policy_tower_dim=2**6,
        logit_dim=2**10,
        value_dim=2**10,
    ) -> None:
        super().__init__()

        (
            our_active_features,
            our_bench_features,
            opp_active_features,
            opp_bench_features,
        ) = group_features

        self.max_traj_split = 64
        self.max_batch_split = 16

        self.pre = ObservationEncoder(
            move_features=move_features,
            our_bench_features=our_bench_features,
            our_active_features=our_active_features,
            opp_bench_features=opp_bench_features,
            opp_active_features=opp_active_features,
        )
        self.body = neural.dot_scaled_linear.SelfDSLTower(
            depth=depth_body,
            row_dims=group_features,
            dot_dim_scale=dot_dim_scale_body,
        )
        self.policy_head = PolicyHead(
            group_features=group_features,
            depth_1=depth_policy_cross,
            depth_2=depth_policy_self,
            dot_dim_scale_1=dot_dim_scale_policy,
            dot_dim_scale_2=dot_dim_scale_policy,
            tower_dim=policy_tower_dim,
            logit_dim=logit_dim,
            move_features=move_features,
            bench_features=group_features[1],
        )
        self.value_head = ValueHead(
            depth=depth_value,
            group_features=group_features,
            dot_dim_scale=dot_dim_scale_value,
            value_dim=value_dim,
        )

    def forward(self, obs_dict: dict):

        T, B, *_ = obs_dict["active_boosts_vol"].shape
        encoded_obs: EncodedObservation = self.pre.forward(obs_dict)
        encoded_obs = EncodedObservation(
            words=[torch.flatten(word, 0, 1) for word in encoded_obs.words],
            moves=torch.flatten(encoded_obs.moves, 0, 1),
        )
        latent_moves = encoded_obs.moves
        latent_words = self.body(encoded_obs.words)

        logits = self.policy_head(latent_moves, latent_words)
        policy = _legal_policy(logits, obs_dict["legal_actions"].flatten(0, 1))
        log_policy = legal_log_policy(logits, obs_dict["legal_actions"].flatten(0, 1))
        value = self.value_head(latent_words)
        predictions = dict(
            policy=policy.view(T, B, 9),
            value=value.view(T, B, 1),
            log_policy=log_policy.view(T, B, 9),
            logits=logits.view(T, B, 9),
        )
        return predictions

    @torch.no_grad()
    def act(self, obs):
        return self(obs)

    @torch.no_grad()
    def inference(
        self, batch: Dict[str, torch.Tensor], valid: Optional[torch.Tensor] = None
    ):
        T, B, *_ = batch["public_dex"].shape
        device = batch["public_dex"].device
        num_batch_splits = math.ceil(B / self.max_batch_split)
        num_traj_splits = math.ceil(T / self.max_traj_split)

        minibatch_max_len = valid.sum(0)
        minibatch_max_len = minibatch_max_len.view(
            num_batch_splits, B // num_batch_splits
        )
        minibatch_max_len = minibatch_max_len.max(-1).values.tolist()

        output_buffer = {
            key: torch.ones(**value)
            for key, value in output_buffer_specs(T, B, device).items()
        }

        predictions = []
        for b in range(num_batch_splits):
            min_iter = math.ceil(minibatch_max_len[b] / self.max_traj_split)
            for t in range(min(min_iter, num_traj_splits)):
                minibatch = {
                    key: value[
                        self.max_traj_split * t : self.max_traj_split * (t + 1),
                        self.max_batch_split * b : self.max_batch_split * (b + 1),
                    ]
                    for key, value in batch.items()
                }
                predictions = self(minibatch)
                for key, value in predictions.items():
                    output_buffer[key][
                        self.max_traj_split * t : self.max_traj_split * (t + 1),
                        self.max_batch_split * b : self.max_batch_split * (b + 1),
                    ][...] = value

        return Inference(*list(output_buffer.values()))

    def backpropagate(
        self,
        batch: Dict[str, torch.Tensor],
        value_targets: List[torch.Tensor],
        policy_targets: List[torch.Tensor],
        has_playeds: List[torch.Tensor],
        valid: torch.Tensor,
        player_id: torch.Tensor,
        legal_actions: torch.Tensor,
        clip: int = 10_000,
        threshold: float = 2.0,
        outside_weight=1,
    ):
        T, B, *_ = batch["public_dex"].shape
        num_batch_splits = math.ceil(B / self.max_batch_split)
        num_traj_splits = math.ceil(T / self.max_traj_split)

        minibatch_max_len = valid.sum(0)
        minibatch_max_len = minibatch_max_len.view(
            num_batch_splits, B // num_batch_splits
        )
        minibatch_max_len = minibatch_max_len.max(-1).values.tolist()

        is_vector = torch.unsqueeze(torch.ones_like(valid), dim=-1)
        importance_sampling_corrections = [is_vector] * 2

        total_value = valid.sum()

        neurd_losses = []
        value_losses = []
        total_losses = []

        for b in range(num_batch_splits):
            min_iter = math.ceil(minibatch_max_len[b] / self.max_traj_split)
            for t in range(min(min_iter, num_traj_splits)):
                minibatch = {
                    key: value[
                        self.max_traj_split * t : self.max_traj_split * (t + 1),
                        self.max_batch_split * b : self.max_batch_split * (b + 1),
                    ]
                    for key, value in batch.items()
                }
                predictions = self(minibatch)

                minibatch_value_target = [
                    value_target[
                        self.max_traj_split * t : self.max_traj_split * (t + 1),
                        self.max_batch_split * b : self.max_batch_split * (b + 1),
                    ]
                    for value_target in value_targets
                ]
                minibatch_has_played = [
                    has_played[
                        self.max_traj_split * t : self.max_traj_split * (t + 1),
                        self.max_batch_split * b : self.max_batch_split * (b + 1),
                    ]
                    for has_played in has_playeds
                ]

                minibatch_policy_target = [
                    policy_target[
                        self.max_traj_split * t : self.max_traj_split * (t + 1),
                        self.max_batch_split * b : self.max_batch_split * (b + 1),
                    ]
                    for policy_target in policy_targets
                ]
                minibatch_valid = valid[
                    self.max_traj_split * t : self.max_traj_split * (t + 1),
                    self.max_batch_split * b : self.max_batch_split * (b + 1),
                ]
                minibatch_player_id = player_id[
                    self.max_traj_split * t : self.max_traj_split * (t + 1),
                    self.max_batch_split * b : self.max_batch_split * (b + 1),
                ]
                minibatch_legal_actions = legal_actions[
                    self.max_traj_split * t : self.max_traj_split * (t + 1),
                    self.max_batch_split * b : self.max_batch_split * (b + 1),
                ]
                minibatch_isc = [
                    isc[
                        self.max_traj_split * t : self.max_traj_split * (t + 1),
                        self.max_batch_split * b : self.max_batch_split * (b + 1),
                    ]
                    for isc in importance_sampling_corrections
                ]

                accumulation_step_weight = minibatch_valid.sum() / total_value
                loss_weight = accumulation_step_weight.item() * outside_weight

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
        total_loss = sum(total_losses).item()

        return value_loss, neurd_loss, total_loss

    @torch.no_grad()
    def step_weights(self, other: "BasketNet", lr: float):
        for self_param, other_param in zip(self.parameters(), other.parameters()):
            grad = self_param - other_param
            self_param.data.sub_(grad * lr)
