import asyncio
import logging

from typing import Any

from neural.replay_buffer import ReplayBuffer

from poke_env import PlayerConfiguration, LocalhostServerConfiguration

from mp.player import DeepNashPlayer
from torch.nn import Module

from torch.jit import trace


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class Worker:
    def __init__(
        self,
        worker_index: int,
        num_games: int,
        num_players: int,
        battle_format: str = "gen8randombattle",
        team: str = "null",
    ):
        self.worker_index = worker_index
        self.num_games = num_games
        self.num_players = num_players
        self.battle_format = battle_format
        self.team = team

    def __repr__(self):
        return f"Worker(idx={self.worker_index},num_players={self.num_players})"


class SelfPlayWorker(Worker):
    def __init__(
        self,
        worker_index: int,
        num_games: int,
        num_players: int,
        battle_format: str = "gen3sampleteamrandbats",
        team: str = None,
        obs_dict=None,
        shared_hash=None,
        name="p",
    ):
        super().__init__(
            worker_index=worker_index,
            num_games=num_games,
            num_players=num_players,
            battle_format=battle_format,
            team=team,
        )
        self.obs_dict = obs_dict
        self.shared_hash = shared_hash
        self.hash = None
        self.name = name

    def __repr__(self):
        return f"Worker(idx={self.worker_index},num_players={self.num_players})"

    def run(self, model: Module, replay_buffer: ReplayBuffer) -> Any:
        """
        Start selfplay between two asynchronous actors
        """

        local_net = trace(model, self.obs_dict, strict=False)
        local_net.eval()

        async def selfplay(
            model: Module, replay_buffer: ReplayBuffer, shared_model=None
        ):
            return await asyncio.gather(
                *[
                    self.actor(
                        self.worker_index * self.num_players + i,
                        model,
                        replay_buffer,
                        shared_model,
                    )
                    for i in range(self.num_players)
                ]
            )

        results = asyncio.run(selfplay(local_net, replay_buffer, model))
        return results

    async def actor(
        self,
        player_index: int,
        model: Module,
        replay_buffer: ReplayBuffer,
        shared_model,
    ) -> Any:
        """
        Asynchronous actor loop, neccesary for communication with websocket environment
        """
        username = self.name + f"-{player_index}"

        player = DeepNashPlayer(
            model=model,
            replay_buffer=replay_buffer,
            player_configuration=PlayerConfiguration(username, None),
            server_configuration=LocalhostServerConfiguration,
            save_replays=True,
            battle_format=self.battle_format,
            team=self.team,
        )

        for _ in range(self.num_games):

            if shared_model is not None:
                if self.shared_hash is not None:
                    with self.shared_hash.get_lock():
                        if self.shared_hash.value != self.hash:
                            model.load_state_dict(shared_model.state_dict())
                            self.hash = self.shared_hash.value

            try:
                await asyncio.wait_for(player.ladder(1), 10 * 60)

            except asyncio.TimeoutError:
                if replay_buffer is not None:
                    replay_buffer.clean(player.battle_tag)

            else:
                reward = player.get_reward()
                if replay_buffer is not None:
                    player.replay_buffer.ammend_reward(
                        player.battle_tag, player.pid, reward
                    )
                    player.replay_buffer.register_done(player.battle_tag)
                player.reset()
