import time
import torch
import threading
import logging

from typing import List, Tuple
from mp.worker import SelfPlayWorker

import torch.multiprocessing as mp

from main import RNaD

logging.disable(logging.CRITICAL)
# TODO


class Eval:
    def __init__(self, tags: List[Tuple[str, int, int]]) -> None:
        self.tags = tags
        self.nets = []
        self.processes: List[mp.Process] = []
        self.workers = []

        for tag in tags:
            directory_name, m, n = tag
            rnad = RNaD(
                directory_name=directory_name,
                net_init_params=
                {
                    "type": "BasketNet4",
                    "depth_body": 2,
                    "depth_policy_cross": 1,
                    "depth_policy_self": 1,
                    "depth_value": 1,
                    "group_features": (2**7, 2**6, 2**7, 2**6),
                    "move_features": 2**4,
                    "dot_dim_scale_body": 1,
                    "dot_dim_scale_value": 1,
                    "dot_dim_scale_policy": 1,
                    "policy_tower_dim": 2**6,
                    "logit_dim": 2**10,
                    "value_dim": 2**10,
                },
            )
            rnad._load_checkpoint(m, n)
            net = rnad.net_target
            net.share_memory()
            net.to(torch.device("cpu"))
            self.nets.append(net)
            # self.nets.append(None)

        self.n_workers = len(tags)

    def _start_workers(self, n_players_per_worker=1):
        for worker_index in range(self.n_workers):
            name='-'.join([str(_) for _ in self.tags[worker_index]])
            print(name)
            worker = SelfPlayWorker(
                worker_index=worker_index,
                num_games=2**20,
                num_players=n_players_per_worker,
                name=name,
            )
            process = self.ctx.Process(
                target=worker.run,
                args=(
                    self.nets[worker_index],
                    None,
                ),
                name=repr(worker),
            )
            self.processes.append(process)
            self.workers.append(worker)

        for process in self.processes:
            process.start()

    def run(self, n_players_per_worker=1):
        self.ctx = mp.get_context('spawn')
        self._start_workers(n_players_per_worker=n_players_per_worker)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    test = Eval([("cleffa", _, 0) for _ in range(4, 8)])
    test.run(n_players_per_worker=1)
