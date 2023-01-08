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
    policy = exp_logits / exp_logits_sum
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
    ):
        super().__init__()
        d_ = 8
        m_id = 8
        m_pp = 4
        ms_ = 32
        s_ = 32
        hp_ = 4  # 8 TODO
        status_ = 4

        self.dex_emb = nn.Embedding(len(POKEDEX_STOI) + 1, d_)

        self.move_id_emb = nn.Embedding(len(MOVES_STOI) + 1, m_id)
        self.move_pp_emb = nn.Embedding(65, m_pp)
        self.move_linear = nn.Linear(
            m_id + m_pp, move_features, bias=True
        )  # cus relu right after
        self.moveset_linear = nn.Linear(move_features, ms_, bias=False)

        self.item_emb = nn.Embedding(len(ITEMS_STOI) + 1, 4)
        self.ability_emb = nn.Embedding(len(ABILITIES_STOI) + 1, 8)

        self.hp_linear = nn.Linear(1, hp_, bias=False)
        self.status_emb = nn.Embedding(8, status_)

        self.stat_linear = nn.Linear(6, s_, bias=False)

        self.dmia_linear = nn.Linear(d_ + ms_ + 4 + 8, 64, bias=False)

        self.weather_linear = nn.Linear(8, 8)
        self.slot_linear = nn.Linear(20, 16, bias=False)

        # self.boosts_linear = nn.Linear(7, 16, bias=False)
        # self.vol_linear = nn.Linear(40, 32, bias=False)
        self.boosts_vol_linear = nn.Linear(171, 48, bias=False)

        # a u v f level

        f_public = 64
        f_private = 96
        f_active = 64
        f_field = 32

        self.public_linear = nn.Linear(64 + hp_ + status_, f_public, bias=False)
        self.private_linear = nn.Linear(64 + s_, f_private, bias=False)
        self.active_linear = nn.Linear(48, f_active, bias=False)
        self.field_linear = nn.Linear(24, f_field, bias=False)

        #
        self.our_active = nn.Linear(
            f_public + f_private + f_active + f_field, our_active_features
        )
        self.our_bench = nn.Linear(f_public + f_private + f_field, our_bench_features)
        self.opp_active = nn.Linear(f_public + f_active + f_field, opp_active_features)
        self.opp_bench = nn.Linear(f_public + f_field, opp_bench_features)

        self.our_active_norm = nn.LayerNorm(our_active_features)
        self.our_bench_norm = nn.LayerNorm(our_bench_features)
        self.opp_active_norm = nn.LayerNorm(opp_active_features)
        self.opp_bench_norm = nn.LayerNorm(opp_bench_features)

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

        enc_dmia = self.dmia_linear(
            torch.cat(
                (
                    pub_priv_dex_emb,
                    enc_all_moveset[..., :18, :],
                    pub_priv_item_emb,
                    pub_priv_ability_emb[..., :18, :],
                ),
                dim=-1,
            )
        )  # (..., 18, 64)

        emb_stats = self.stat_linear(obs_dict["private_stats"])  # (..., 6, 32)

        emb_weather = self.weather_linear(obs_dict["weather"])
        emb_slots = self.slot_linear(obs_dict["slot"])  # (..., 2, 16)

        emb_boosts_vol = self.boosts_vol_linear(obs_dict["active_boosts_vol"])

        # # # #

        enc_public = self.public_linear(
            torch.cat(
                (
                    enc_dmia[..., :12, :],
                    self.hp_linear(obs_dict["public_hp"]),
                    self.status_emb(obs_dict["public_status"][..., 0]),
                ),
                dim=-1,
            )
        )  # (..., 12, 64)
        enc_private = torch.cat(
            (emb_stats, enc_dmia[..., 12:, :]), dim=-1
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
            torch.cat((emb_boosts_vol,), dim=-1)
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

        return EncodedObservation(
            words=[our_active, our_bench, opp_active, opp_bench],
            moves=all_moves_linear[..., 18, :, :],  # (... 4, m_)
        )


class PolicyHead(torch.nn.Module):
    def __init__(
        self,
        col_group_features=(2**7, 2**6),
        row_group_features=(2**4 + 2**7, 2**6),
        inner_feature_dict=None,
        outer_dim=2**8,
        num_heads=1,
        dropout=0.0,
        bias=True,
        batch_first=True,
    ) -> None:

        super().__init__()

        self.m_groups = len(row_group_features)
        self.n_groups = len(col_group_features)

        if inner_feature_dict is None:
            inner_feature_dict = {}
            for a in range(self.m_groups):
                f = row_group_features[a]
                for b in range(self.n_groups):
                    g = col_group_features[b]
                    inner_feature_dict[(a, b)] = min(f, g)

        self.cross_attn = torch.nn.ParameterDict(
            {
                f"({a}, {b})": MultiheadAttention(
                    embed_dim=row_group_features[a],
                    inner_dim=inner_feature_dict[(a, b)],
                    outer_dim=inner_feature_dict[(a, b)],
                    kdim=col_group_features[b],
                    vdim=col_group_features[b],
                    output_dim=outer_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                    batch_first=batch_first,
                )
                for a, b in neural.attention.prod(self.m_groups, self.n_groups)
            }
        )
        self.cross_attn_norms = torch.nn.ParameterList(
            [torch.nn.LayerNorm((outer_dim,)) for a in range(self.m_groups)]
        )

        # self.self_attn = neural.attention.GroupedMultiheadSelfAttention(
        #     group_features=(outer_dim, outer_dim),
        #     output_features=(outer_dim, outer_dim),
        #     num_heads=num_heads,
        #     batch_first=batch_first,
        #     dropout=dropout,
        #     bias=bias,
        # )

        self.self_attn = neural.attention.GroupedTransformerEncoderLayer(
            group_features=(
                outer_dim,
                outer_dim,
            ),
            feed_forward_features=(outer_dim, outer_dim),
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            num_heads=num_heads,
        )

        self.logit_linear_1 = torch.nn.Linear(outer_dim, 2 * outer_dim)
        self.logit_linear_2 = torch.nn.Linear(2 * outer_dim, 1)

    def forward(self, words: List[torch.Tensor], cross_words: List[torch.Tensor]):
        words_ = []
        for a in range(self.m_groups):
            n, batch, _ = words[a].shape
            # Duplicate code to SelfAttention...

            group_outputs = torch.empty(
                size=(n, batch, 256, self.n_groups), device=words[a].device
            )
            group_weights = torch.empty(
                size=(n, batch, 1, self.n_groups), device=words[a].device
            )

            for b in range(self.n_groups):
                layer = self.cross_attn[f"({a}, {b})"]
                attn_outputs, logits = layer(words[a], cross_words[b], cross_words[b])
                group_outputs[..., b] = attn_outputs
                group_weights[..., 0, b] = torch.sum(torch.exp(logits), dim=-1)
            group_weights = torch.nn.functional.normalize(group_weights, dim=-1, p=1)

            group_outputs *= group_weights
            word = torch.sum(group_outputs, dim=-1)
            word = self.cross_attn_norms[a](word)
            words_.append(word)

        words_ = self.self_attn(words_)
        words_ = torch.cat(words_, dim=-2)
        words_ = torch.relu(self.logit_linear_1(words_))
        logits = self.logit_linear_2(words_)[..., 0]
        return logits


class ValueHead(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        attn_features = 2**6
        self.attention = neural.attention.GroupedMultiheadSelfAttention(
            group_features=(2**7, 2**6, 2**7, 2**6),
            output_features=(attn_features,) * 4,
        )
        inner_features = 2**8
        self.linear_inner = torch.nn.Linear(attn_features * 12, inner_features)
        self.linear_outer = torch.nn.Linear(inner_features, 1)

    def forward(self, words):
        words_ = torch.relu(
            torch.flatten(torch.cat(self.attention(words), dim=-2), -2, -1)
        )
        inner = torch.relu(self.linear_inner(words_))
        value = self.linear_outer(inner)
        return value


class BasketNet2(nn.Module):
    def __init__(
        self,
        depth=3,
        group_features=(2**7, 2**6, 2**7, 2**6),
        move_features=16,
        inner_feature_dict=None,
        num_heads=1,
        feed_forward_features=(2**7, 2**6, 2**7, 2**6),
        dropout=0.1,
        bias=True,
        batch_first=True,
    ) -> None:
        super().__init__()

        (
            our_active_features,
            our_bench_features,
            opp_active_features,
            opp_bench_features,
        ) = group_features

        self.max_traj_split = 32
        self.max_batch_split = 16

        self.pre = ObservationEncoder(
            move_features=move_features,
            our_bench_features=our_bench_features,
            our_active_features=our_active_features,
            opp_bench_features=opp_bench_features,
            opp_active_features=opp_active_features,
        )
        self.body = neural.dot_scaled_linear.SelfDSLTower(
            depth=4,
            row_dims=group_features,
            dot_dim_scale=2,
            dropout=dropout,
        )
        self.policy_head = PolicyHead(
            col_group_features=group_features[-2:],
            row_group_features=(
                move_features + our_active_features,
                our_bench_features,
            ),
            batch_first=batch_first,
            bias=bias,
            dropout=dropout,
            num_heads=1,
        )
        self.value_head = ValueHead()

    def forward(self, obs_dict: dict, as_list: bool = False):

        T, B, *_ = obs_dict["active_boosts_vol"].shape
        encoded_obs: EncodedObservation = self.pre.forward(obs_dict)
        encoded_obs = EncodedObservation(
            words=[torch.flatten(word, 0, 1) for word in encoded_obs.words],
            moves=torch.flatten(encoded_obs.moves, 0, 1),
        )
        latent_moves = encoded_obs.moves
        latent_words = self.body(encoded_obs.words)
        (
            latent_our_active,
            latent_our_bench,
            latent_opp_active,
            latent_opp_bench,
        ) = latent_words
        latent_active_moves = torch.cat(
            [latent_moves, latent_our_active.expand(T * B, 4, 2**7)], dim=-1
        )

        logits = self.policy_head(
            [latent_active_moves, latent_our_bench],
            [latent_opp_active, latent_opp_bench],
        )
        policy = _legal_policy(logits, obs_dict["legal_actions"].flatten(0, 1))
        log_policy = legal_log_policy(logits, obs_dict["legal_actions"].flatten(0, 1))
        value = self.value_head(latent_words)
        predictions = dict(
            policy=policy.view(T, B, 9),
            value=value.view(T, B, 1),
            log_policy=log_policy.view(T, B, 9),
            logits=logits.view(T, B, 9),
        )

        if as_list:
            return Inference(*list(predictions.values()))
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

                loss_v = get_loss_v(
                    [predictions["value"]] * 2,
                    minibatch_value_target,
                    minibatch_has_played,
                )

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

                loss_nerd = get_loss_nerd(
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

                accumulation_step_weight = minibatch_valid.sum() / total_value

                loss = loss_v + loss_nerd
                neurd_losses.append(loss_nerd)
                value_losses.append(loss_v)
                total_losses.append(loss)

                loss *= accumulation_step_weight.item()
                loss *= outside_weight
                loss.backward()

        neurd_loss = (sum(neurd_losses) / len(neurd_losses)).item()
        value_loss = (sum(value_losses) / len(value_losses)).item()
        total_loss = (sum(total_losses) / len(total_losses)).item()

        return value_loss, neurd_loss, total_loss

    @torch.no_grad()
    def step_weights(self, other: "BasketNet2", lr: float):
        for self_param, other_param in zip(self.parameters(), other.parameters()):
            grad = self_param - other_param
            self_param.data.sub_(grad * lr)
