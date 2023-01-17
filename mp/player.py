import torch
import torch.nn.functional as F

from typing import Dict, List, NamedTuple, Literal, Optional, Tuple

from poke_env.data import (
    POKEDEX_STOI,
    POKEDEX_ITOS,
    ABILITIES_STOI,
    MOVES_STOI,
    MOVES_ITOS,
    ITEMS_STOI,
    to_id_str,
    MOVE_CUTOFF,
    POKEDEX_CUTOFF,
)

from poke_env.environment import (
    AbstractBattle,
    Pokemon,
    Move,
    Effect,
    SideCondition,
    Weather,
)

from poke_env.player.player import Player
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

from rnad.finetuning import FineTuning
from neural.buffers import ReplayBuffer


class ObservationTensor(NamedTuple):
    # public stuff
    public_dex: torch.Tensor  # (12, 1) int
    public_hp: torch.Tensor
    public_status: torch.Tensor
    public_moves: torch.Tensor  # (12, 4, 2) idx, curr_pp
    public_item: torch.Tensor  # (12, 1)
    public_ability: torch.Tensor  # (12, 1)
    # private stuff
    private_stats: torch.Tensor  # (6, 6,)
    private_dex: torch.Tensor  # (6, 1)
    private_moves: torch.Tensor  # (6, 4, 2) idx, curr_pp
    private_item: torch.Tensor  # (6, 1)
    private_ability: torch.Tensor  # (6, 1)
    # field
    weather: torch.Tensor  # (1, 8)
    slot: torch.Tensor  # (2, n)
    # active
    active_boosts_vol: torch.Tensor  # (2, 171)
    active_moves: torch.Tensor  # (1, 4, 2)
    legal_actions: torch.Tensor
    move_index: torch.Tensor
    switch_index: torch.Tensor

    def detach(self):
        obs_dict: Dict[str, torch.Tensor] = self._asdict()
        return ObservationTensor(
            **{key: value.cpu().detach() for key, value in obs_dict.items()}
        )


class HistoryStep(NamedTuple):
    turn: int
    step: int
    data: List[str]


class Observation:
    ACTIONS = {"|switch|", "|move|"}

    def __init__(self, battle: AbstractBattle):
        self.battle = battle

        self.pid = 0 if battle.player_role == "p1" else 1
        self.battle_tag = battle.battle_tag

    def get(self) -> ObservationTensor:

        our_pokemon: List[Pokemon] = sorted(
            self.battle.team.values(),
            key=lambda pokemon: pokemon.active,
            reverse=True,
        )
        our_pokemon += [None for _ in range(6 - len(our_pokemon))]
        opp_pokemon: List[Pokemon] = sorted(
            self.battle.opponent_team.values(),
            key=lambda pokemon: pokemon.active,
            reverse=True,
        )
        opp_pokemon += [None for _ in range(6 - len(opp_pokemon))]

        all_pokemon: List[Pokemon] = our_pokemon + opp_pokemon

        public_dex = torch.tensor(
            [
                0
                if pokemon is None
                else POKEDEX_STOI[pokemon.species]
                if pokemon.revealed
                else 0
                for pokemon in all_pokemon
            ]
        ).unsqueeze(-1)

        public_hp = torch.tensor(
            [
                1.0 if pokemon is None else pokemon.current_hp_fraction
                for pokemon in all_pokemon
            ]
        ).unsqueeze(-1)

        public_status = torch.tensor(
            [
                0
                if pokemon is None
                else 0
                if pokemon.status is None
                else pokemon.status.value
                for pokemon in all_pokemon
            ]
        ).unsqueeze(-1)

        public_move_list = []
        for pokemon in all_pokemon:
            if pokemon is None:
                public_move_list.append([[0, 0]] * 4)
                continue
            move_vectors = [
                self.get_public_move_vector(move)
                for move in sorted(
                    pokemon.moves.values(),
                    key=lambda move: move.id,
                )
                if (move.current_pp / move.max_pp) < 1
            ]
            for _ in range(4 - len(move_vectors)):
                unk_move = [0, 0]
                move_vectors.append(unk_move)
            public_move_list.append(move_vectors)
        public_moves = torch.tensor(public_move_list)

        private_move_list = []
        for pokemon in our_pokemon:
            if pokemon is None:
                private_move_list.append([[0, 0]] * 4)
                continue
            move_vectors = [
                self.get_public_move_vector(move)
                for move in sorted(
                    pokemon.moves.values(),
                    key=lambda move: move.id,
                )
            ]
            for _ in range(4 - len(move_vectors)):
                unk_move = [0, 0]
                move_vectors.append(unk_move)
            private_move_list.append(move_vectors)
        private_moves = torch.tensor(private_move_list)

        active_move_list = []
        legal_moves = []
        for pokemon in our_pokemon[:1]:
            if pokemon is None:
                active_move_list.append([[0, 0]] * 4)
                legal_moves = [0, 0, 0, 0]
                move_index = [1, 1, 1, 1]
                continue
            move_vectors = [
                self.get_public_move_vector(move)
                for move in sorted(
                    self.battle.available_moves,
                    key=lambda move: move.id,
                )
            ]
            legal_moves = [1 for vector in move_vectors]
            for _ in range(4 - len(move_vectors)):
                unk_move = [0, 0]
                move_vectors.append(unk_move)
                legal_moves.append(0)
            active_move_list.append(move_vectors)
        active_moves = torch.tensor(active_move_list)
        legal_moves = torch.tensor(legal_moves)
        move_index = active_moves[0, :, 0]

        private_item = torch.tensor(
            [
                0 if pokemon is None else ITEMS_STOI.get(pokemon.item, 0)
                for pokemon in our_pokemon
            ]
        ).unsqueeze(dim=-1)
        public_item = torch.tensor(
            [
                0 if pokemon is None else ITEMS_STOI.get(pokemon.public_item, 0)
                for pokemon in all_pokemon
            ]
        ).unsqueeze(dim=-1)
        public_ability = torch.tensor(
            [
                0 if pokemon is None else ABILITIES_STOI.get(pokemon.ability, 0)
                for pokemon in all_pokemon
            ]
        ).unsqueeze(-1)

        private_stats = torch.tensor(
            [
                [0] * 6
                if pokemon is None
                else [pokemon.max_hp] + list(pokemon.stats.values())
                for pokemon in our_pokemon
            ],
            dtype=torch.float,
        )
        private_dex = torch.tensor(
            [
                0 if pokemon is None else POKEDEX_STOI[pokemon.species]
                for pokemon in our_pokemon
            ]
        ).unsqueeze(-1)

        switch_index = torch.tensor(
            [
                0
                if pokemon is None
                or pokemon not in self.battle.available_switches
                or self.battle.trapped
                else POKEDEX_STOI[pokemon.species]
                for pokemon in our_pokemon[1:]
            ]
        )
        legal_switches = switch_index != 0
        legal_actions = torch.cat([legal_moves, legal_switches], dim=-1).to(torch.bool)

        private_ability = public_ability[:6]

        weather = self.get_weather().unsqueeze(dim=-2)
        slot = torch.stack(
            (self.get_side_conditions("player"), self.get_side_conditions("opponent")),
            dim=-2,
        ).to(torch.float)
        active_boosts_vol = torch.stack(
            (self.get_boosts_volatile("player"), self.get_boosts_volatile("opponent")),
            dim=-2,
        ).to(torch.float)
        obs = ObservationTensor(
            public_dex=public_dex,
            public_hp=public_hp,
            public_status=public_status,
            public_moves=public_moves,
            public_item=public_item,
            public_ability=public_ability,
            private_stats=private_stats,
            private_dex=private_dex,
            private_moves=private_moves,
            private_item=private_item,
            private_ability=private_ability,
            weather=weather,
            slot=slot,
            active_boosts_vol=active_boosts_vol,
            active_moves=active_moves,
            legal_actions=legal_actions,
            move_index=move_index,
            switch_index=switch_index,
        )
        return obs

    def get_public_move_vector(self, move: Move) -> torch.Tensor:
        move_vector = [
            MOVES_STOI[move.id],
            max(0, move.current_pp),
        ]
        return move_vector

    def get_boosts_volatile(
        self, player: Literal["player", "opponent"]
    ) -> torch.Tensor:
        active: Pokemon
        if player == "player":
            active = self.battle.active_pokemon
        else:
            active = self.battle.opponent_active_pokemon

        boosts_vector = torch.tensor(
            [
                active.boosts["accuracy"],
                active.boosts["atk"],
                active.boosts["def"],
                active.boosts["evasion"],
                active.boosts["spa"],
                active.boosts["spd"],
                active.boosts["spe"],
            ],
            dtype=torch.float32,
        )
        vol_status_vector = torch.tensor(
            [active.effects.get(effect, 0) for effect in Effect],
            dtype=torch.float32,
        )
        return torch.cat(
            (
                boosts_vector,
                vol_status_vector,
            ),
            dim=-1,
        )

    def get_side_conditions(
        self, player: Literal["player", "opponent"]
    ) -> torch.Tensor:
        if player == "player":
            side_conditions = self.battle.side_conditions
        elif player == "opponent":
            side_conditions = self.battle.opponent_side_conditions
        side_condt_vector = torch.tensor(
            [side_conditions.get(sc, 0) for sc in SideCondition],
            dtype=torch.long,
        )
        return side_condt_vector

    def get_weather(self) -> torch.Tensor:
        weather_vector = torch.tensor(
            [
                min(8, self.battle.turn - self.battle.weather[weather])
                if weather in self.battle.weather
                else 0
                for weather in Weather
            ],
            dtype=torch.float,
        )
        return weather_vector


class DeepNashPlayer(Player):
    def __init__(
        self,
        model: torch.nn.Module,
        replay_buffer: Optional[ReplayBuffer] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model = model
        self.replay_buffer = replay_buffer
        self.finetune = FineTuning()

        self.pid = None
        self.battle_tag = None

    def choose_move(self, battle: AbstractBattle) -> Tuple[BattleOrder, str]:

        if battle.turn >= 300:
            print("Forfeit", self.battle_tag)
            return ForfeitBattleOrder(), ""
        observation = Observation(battle)
        self.pid = observation.pid
        self.battle_tag = observation.battle_tag
        obs = observation.get()
        obs_dict = obs._asdict()

        for key, value in obs_dict.items():
            obs_dict[key] = value.view(1, 1, *value.shape)

        # model_ = torch.jit.trace(self.model, obs_dict, strict=False)
        # torch.save(obs_dict, 'obs_dict.tar')
        with torch.no_grad():
            prediction = self.model(obs_dict)

        policy = prediction["policy"].view(-1)

        pprocessed_policy = policy
        action_index = torch.multinomial(pprocessed_policy, 1)

        action_oh = F.one_hot(action_index, policy.shape[-1]).view(-1)
        move_switch_index = torch.cat([obs.move_index, obs.switch_index])

        policy_text = "".join(
            [
                f"{MOVES_ITOS[idx.item()]}: {int(p.item() * 100)} "
                if _ < 4
                else f"{POKEDEX_ITOS[idx.item()]}: {int(p.item() * 100)} "
                for _, (p, idx, l) in enumerate(
                    zip(
                        prediction["policy"].view(-1),
                        move_switch_index,
                        obs.legal_actions,
                    )
                )
                if l
            ]
        )
        policy_text += f'VALUE: {round(prediction["value"].item(), 3)} '

        if self.replay_buffer is not None:
            obs = obs.detach()
            step = {
                "player_id": observation.pid,
                "action_oh": action_oh,
                "policy": policy,
                **obs._asdict(),
            }
            step.pop("move_index")
            step.pop("switch_index")
            self.replay_buffer.store_sample(observation.battle_tag, step)

        try:

            if action_index < 4:
                action_id = MOVES_ITOS[move_switch_index[action_index.item()].item()]
                action = Move(action_id)
            else:
                action_id = POKEDEX_ITOS[move_switch_index[action_index.item()].item()]
                action = Pokemon(species=action_id)
            policy_text += f"ACTION: {action_id}"
        except:
            return self.choose_random_move(battle), "Used Random Action"
        else:
            return BattleOrder(action), policy_text

    def get_reward(self) -> float:
        return 1 if self.battles[self.battle_tag].won else -1

    def reset(self):
        self.pid = None
        self.battle_tag = None
