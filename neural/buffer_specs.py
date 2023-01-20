import torch

from copy import deepcopy
from typing import Any, Dict, List, Sequence, Union
from poke_env.data import MOVE_CUTOFF, POKEDEX_CUTOFF

request = Dict[str, Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]]
buffer_specs = Dict[str, Dict[str, Union[Sequence[int], torch.dtype]]]

MAX_T = 664
POLICY_SIZE = 9
# POLICY_SIZE = 10

representation_input_buffer_specs = dict(
    # Public
    public_dex=dict(
        size=(MAX_T, 12, 1),
        dtype=torch.int,
    ),
    public_hp=dict(
        size=(MAX_T, 12, 1),
        dtype=torch.float,
    ),
    public_status=dict(
        size=(MAX_T, 12, 1),
        dtype=torch.int,
    ),
    public_moves=dict(
        size=(MAX_T, 12, 4, 2),
        dtype=torch.int,
    ),
    public_item=dict(
        size=(MAX_T, 12, 1),
        dtype=torch.int,
    ),
    public_ability=dict(
        size=(MAX_T, 12, 1),
        dtype=torch.int,
    ),
    # Private
    private_stats=dict(
        size=(MAX_T, 6, 6),
        dtype=torch.float,
    ),
    private_dex=dict(
        size=(MAX_T, 6, 1),
        dtype=torch.int,
    ),
    private_moves=dict(
        size=(MAX_T, 6, 4, 2),
        dtype=torch.int,
    ),
    private_item=dict(
        size=(MAX_T, 6, 1),
        dtype=torch.int,
    ),
    private_ability=dict(
        size=(MAX_T, 6, 1),
        dtype=torch.int,
    ),
    # Active
    active_moves=dict(
        size=(MAX_T, 1, 4, 2),
        dtype=torch.int,
    ),
    # active_ability=dict(
    #     size=(MAX_T, 2, 111),
    #     dtype=torch.int,
    # ),
    # active_boosts=dict(
    #     size=(MAX_T, 2, 7),
    #     dtype=torch.int,
    # ),
    # active_vol=dict(
    #     size=(MAX_T, 2, 111),
    #     dtype=torch.int,
    # ),
    # active_types=dict(
    #     size=(MAX_T, 2, 2),
    #     dtype=torch.int,
    # ),
    active_boosts_vol=dict(
        size=(MAX_T, 2, 171),
        dtype=torch.float,
    ),
    # Field
    weather=dict(
        size=(MAX_T, 1, 8),
        dtype=torch.float,
    ),
    slot=dict(
        size=(MAX_T, 2, 20),
        dtype=torch.float,
    ),
    # Legal
    legal_actions=dict(
        size=(MAX_T, POLICY_SIZE),
        dtype=torch.bool,
    ),
)

prediction_output_buffer_specs = dict(
    policy=dict(
        size=(MAX_T, POLICY_SIZE),
        dtype=torch.float32,
    ),
)

reward_valid_specs = dict(
    rewards=dict(
        size=(MAX_T, 2),
        dtype=torch.float32,
    ),
    valid=dict(
        size=(MAX_T,),
        dtype=torch.bool,
    ),
    action_oh=dict(
        size=(MAX_T, POLICY_SIZE),
        dtype=torch.long,
    ),
    player_id=dict(
        size=(MAX_T,),
        dtype=torch.long,
    ),
)

replay_buffer_specs = deepcopy(representation_input_buffer_specs)
replay_buffer_specs.update(prediction_output_buffer_specs)
replay_buffer_specs.update(reward_valid_specs)