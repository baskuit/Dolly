import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

import time
import torch
import wandb
import logging
import threading

from typing import List

import torch.multiprocessing as mp

from torch import nn
from torch import optim

from neural.basket import BasketNet
from neural.basket2 import BasketNet2
from neural.basket3 import BasketNet3
from neural.basket4 import BasketNet4
from neural.buffer_specs import representation_input_buffer_specs
from neural.buffers import ReplayBuffer

from rnad.finetuning import FineTuning
from rnad.vtrace import v_trace, _player_others

from mp.worker import SelfPlayWorker
from random import randint


class RNaD:
    def __init__(
        self,
        format="gen3randombattle",
        learn_device=torch.device("cuda"),
        actor_device=torch.device("cpu"),
        eta=0.2,
        lr=5 * 10**-5,
        beta=2,
        neurd_clip=10**3,
        grad_clip=10**3,
        b1_adam=0,
        b2_adam=0.999,
        epsilon_adam=10**-8,
        gamma_averaging=0.001,
        roh_bar=1,
        c_bar=1,
        batch_size=3 * 2**8,
        epsilon_threshold=0.03,
        n_discrete=32,
        directory_name=None,
        net_init_params=None,
        gamma_vtrace=1,
        replay_buffer_size=None,
        buffer_retain=0,
        bounds=None,
        delta_m=None,
        desc="",
        wandb=False,
    ):
        if bounds is None:
            bounds = (128,)
        if delta_m is None:
            delta_m = (100,)
        self._assert_bounds(bounds, delta_m)

        self.bounds = bounds
        self.delta_m = delta_m
        self.format = format
        self.eta = eta
        self.lr = lr
        self.beta = beta
        self.neurd_clip = neurd_clip
        self.grad_clip = grad_clip
        self.b1_adam = b1_adam
        self.b2_adam = b2_adam
        self.epsilon_adam = epsilon_adam
        self.gamma_averaging = gamma_averaging
        self.rho_bar = roh_bar
        self.c_bar = c_bar
        self.batch_size = batch_size
        self.epsilon_threshold = epsilon_threshold
        self.n_discrete = n_discrete
        self.gamma_vtrace = gamma_vtrace
        self.desc = desc
        self.wandb = wandb

        if directory_name is None:
            directory_name = str(int(time.time()))
        self.directory_name = directory_name

        if net_init_params is None:
            net_init_params = {"type": "BasketNet"}
        self.net_init_params = net_init_params

        self.saved_params = list(self.__dict__.keys())
        # Everything above this is saved and loaded

        if replay_buffer_size is None:
            replay_buffer_size = batch_size
        self.replay_buffer_size = int(replay_buffer_size)

        self.buffer_retain = buffer_retain

        self.learn_device = learn_device
        self.actor_device = actor_device

        saved_runs_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "saved_runs"
        )
        if not os.path.exists(saved_runs_dir):
            os.mkdir(saved_runs_dir)
        self.directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "saved_runs", directory_name
        )

        self.workers: List[SelfPlayWorker] = []
        self.processes: List[mp.Process] = []

        self.finetune = FineTuning()
        self.total_steps = 0

        self.half_trajectory_counter = 0

        self.loss_value = {}
        self.loss_neurd = {}
        self.avg_traj_len = {}

    def _assert_bounds(self, bounds, delta_m):
        for _, __ in zip(bounds, bounds[1:]):
            assert __ > _
        assert len(bounds) == len(delta_m)

    def _get_update_info(self) -> tuple[bool, int]:
        bounding_indices = [_ for _, bound in enumerate(self.bounds) if bound > self.m]
        if not bounding_indices:
            return False, 0
        idx = min(bounding_indices)
        return True, self.delta_m[idx]

    def _assert_schedule(self, bounds, delta_m):
        self._assert_bounds(self.bounds, self.delta_m)
        bounding_indices = [_ for _, bound in enumerate(bounds) if bound > self.m]
        if not bounding_indices:
            return False
        idx = min(bounding_indices)
        for _ in range(idx):
            assert self.bounds[_] == bounds[_]
            assert self.delta_m[_] == delta_m[_]
        # assert self.delta_m[idx] == delta_m[idx]
        # assert self.delta_m[idx] > self.n
        # The schedules can be swapped
        # self. is the new sched. provided by initializer
        # loaded/arg sched. is assumed to be correct

    def _initialize(self):

        self.ctx = mp.get_context("spawn")
        self.online_hash = self.ctx.Value("i", 0)

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        updates = [
            int(os.path.relpath(f.path, self.directory))
            for f in os.scandir(self.directory)
            if f.is_dir()
        ]

        self.obs_dict = torch.load("obs_dict.tar")
        self.obs_dict = {
            key: value.detach().clone() for key, value in self.obs_dict.items()
        }

        if not updates:
            print("Initializing R-NaD run: {}".format(self.directory_name))

            params_dict = {key: self.__dict__[key] for key in self.saved_params}
            torch.save(params_dict, os.path.join(self.directory, "params"))

            os.mkdir(os.path.join(self.directory, "0"))
            self.net_offline = self._new_net()
            self.net_offline.train()
            self.net_online = self._new_net(
                self.net_offline.state_dict(), device=self.actor_device
            )
            self.net_target = self._new_net(self.net_offline.state_dict())
            self.net_reg = self._new_net(self.net_offline.state_dict())
            self.net_reg_ = self._new_net(self.net_offline.state_dict())
            self.optimizer = torch.optim.Adam(
                self.net_offline.parameters(),
                lr=self.lr,
                betas=[self.b1_adam, self.b2_adam],
                eps=self.epsilon_adam,
            )

            self.optimizer = optim.Adam(
                self.net_offline.parameters(),
                lr=self.lr,
                betas=(self.b1_adam, self.b2_adam),
                eps=self.epsilon_adam,
            )

            self.m = 0
            self.n = 0
            self._save_checkpoint()

        else:
            print("Resuming R-NaD run: {}".format(self.directory_name))

            self.m = max(updates)
            last_update = os.path.join(self.directory, str(self.m))
            checkpoints = [
                int(os.path.relpath(f.path, last_update))
                for f in os.scandir(last_update)
                if not f.is_dir()
            ]
            self.n = max(checkpoints)

            params_dict: dict = torch.load(os.path.join(self.directory, "params"))

            for key, value in params_dict.items():
                init_value = self.__dict__[key]
                if key == "directory_name":
                    params_dict[key] = init_value
                    continue
                if key == "delta_m" or key == "bounds":
                    try:
                        self._assert_schedule(
                            params_dict["bounds"], params_dict["delta_m"]
                        )
                        # check that init sched. is compatible with/ extends saved one
                    except AssertionError:
                        print(
                            "Provided schedule inconsitent with saved one, defaulting to saved"
                        )
                    else:
                        print("Checks passed. Using init schedule")
                        params_dict[key] = init_value
                        continue
                if key == "replay_buffer_size":
                    continue
                if key == "desc":
                    if init_value:
                        params_dict[key] = init_value
                        continue
                if value != init_value:
                    print(
                        f"saved param: {key} differed from init, using saved: {value}"
                    )
                self.__dict__[key] = value
            torch.save(params_dict, os.path.join(self.directory, "params"))

            self._load_checkpoint(self.m, self.n)
            self._load_logs()

        self.replay_buffer = ReplayBuffer(
            self.ctx,
            num_buffers=int(self.replay_buffer_size),
            device=torch.device("cpu"),
            retain=self.buffer_retain,
        )
        print(f"Initialized buffer with size: {self.replay_buffer_size}")

        if self.wandb:
            wandb.init(
                # resume=bool(updates),
                resume=True,
                project="Murkrow",
            )

    def _new_net(self, state_dict=None, device=None) -> nn.Module:
        # e.g. {'type':'FCResNet', 'tower_length'=8}
        if self.net_init_params["type"] == "BasketNet":
            t = BasketNet
        if self.net_init_params["type"] == "BasketNet2":
            t = BasketNet2
        if self.net_init_params["type"] == "BasketNet3":
            t = BasketNet3
        if self.net_init_params["type"] == "BasketNet4":
            t = BasketNet4
        net_params = {
            key: value for key, value in self.net_init_params.items() if key != "type"
        }
        net_ = t(**net_params)
        if not state_dict is None:
            net_.load_state_dict(state_dict)
        net_.eval()
        if device is None:
            device = self.learn_device
        net_.to(device)
        print(
            f'Init {self.net_init_params["type"] } net with {sum([_.numel() for _ in net_.parameters() if _.requires_grad])} params'
        )
        return net_

    def _save_checkpoint(self):
        saved_dict = {
            "total_steps": self.total_steps,
            "net_offline": self.net_offline.state_dict(),
            "net_online": self.net_online.state_dict(),
            "net_target": self.net_target.state_dict(),
            "net_reg": self.net_reg.state_dict(),
            "net_reg_": self.net_reg_.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if not os.path.exists(os.path.join(self.directory, str(self.m))):
            os.mkdir(os.path.join(self.directory, str(self.m)))
        torch.save(saved_dict, os.path.join(self.directory, str(self.m), str(self.n)))

    def _load_checkpoint(self, m, n):
        saved_dict = torch.load(os.path.join(self.directory, str(m), str(n)))
        self.total_steps = saved_dict["total_steps"]
        self.net_offline = self._new_net(saved_dict["net_offline"])
        self.net_offline.train()
        self.net_online = self._new_net(
            saved_dict["net_online"], device=self.actor_device
        )
        self.net_target = self._new_net(saved_dict["net_target"])
        self.net_reg = self._new_net(saved_dict["net_reg"])
        self.net_reg_ = self._new_net(saved_dict["net_reg_"])
        self.optimizer = torch.optim.Adam(
            self.net_offline.parameters(),
            lr=self.lr,
            betas=[self.b1_adam, self.b2_adam],
            eps=self.epsilon_adam,
        )
        self.optimizer.load_state_dict(saved_dict["optimizer"])

    def _save_logs(self):
        try:
            saved_logs = {
                "loss_value": self.loss_value,
                "loss_neurd": self.loss_neurd,
                "avg_traj_len": self.avg_traj_len,
            }
            torch.save(saved_logs, os.path.join(self.directory, "logs"))
        except:
            print("_save_logs failed")

    def _load_logs(self):
        try:
            saved_logs = torch.load(os.path.join(self.directory, "logs"))
            for key, value in saved_logs.items():
                self.__dict__[key] = value
        except:
            print("_load_logs failed")

    # def _cosine(self, words):
    #     words = torch.nn.functional.normalize(words[:, :, 1:6, :], p=2, dim=-1)
    #     words_transposed = torch.transpose(words, -1, -2)
    #     dot_product = torch.matmul(words, words_transposed)
    #     means = (torch.sum(dot_product, dim=[-1, -2]) - 5) / 20
    #     return means

    def _learn(
        self,
        n_chunks=8,
        checkpoint_mod=10,
        log_mod=1,
        first_step=True,
    ):

        alpha_lambda = lambda n, delta_m: 1 if n > delta_m / 2 else n * 2 / delta_m

        while True:

            mayResume, delta_m = self._get_update_info()
            if not mayResume:
                return
            print(f"m: {self.m}, n: {self.n}")
            if self.n % checkpoint_mod == 0 and not first_step:
                print("Saving checkpoint")
                self._save_checkpoint()
                print("Done.")
            alpha = alpha_lambda(self.n, delta_m)

            batch_wait_start = time.perf_counter()
            batches = self.replay_buffer.get_batches(self.batch_size, n_chunks=n_chunks)
            batch_retrieval_time = time.perf_counter() - batch_wait_start
            print(f"batch retrieved in {int(batch_retrieval_time)} sec.")

            print(f"CUDA Mem Alloc'd: {int(torch.cuda.memory_allocated() / 2**20)}")
            print(f"CUDA Max Mem: {int(torch.cuda.max_memory_allocated() / 2**20)}")
            torch.cuda.reset_accumulated_memory_stats()
            free_queue_size = self.replay_buffer.free_queue.qsize()
            full_queue_size = self.replay_buffer.full_queue.qsize()
            print(f"Free Queue size: {free_queue_size}")
            print(f"Full Queue size: {full_queue_size}")
            print(f"combined: {free_queue_size + full_queue_size}")

            t_effs = [
                torch.mean(
                    torch.sum(batch["valid"], dim=0).to(torch.float), dim=0
                ).item()
                for batch in batches
            ]
            print("t_effs: ", t_effs)
            learn_start_time = time.time()

            valid_total = sum([torch.sum(batch["valid"]).item() for batch in batches])
            print(f"Valid total: {valid_total}")
            print(f"Speed: {valid_total / batch_retrieval_time}")

            (
                weight_total,
                value_loss_,
                neurd_loss_,
                total_loss_,
                avg_traj_len_,
                actor_learner_kld_,
                entropy_,
                entropy_target_,
            ) = (0, 0, 0, 0, 0, 0, 0, 0)

            for chunk in range(n_chunks):

                batch = batches[chunk]
                batch = {
                    key: value.to(self.learn_device)
                    for key, value in batch.items()
                    if torch.is_tensor(value)
                }

                player_id = batch["player_id"]
                action_oh = batch["action_oh"]
                policy = batch["policy"]
                rewards = batch["rewards"]
                valid = batch["valid"]
                valid_sub_total = torch.sum(batch["valid"]).item()
                outside_weight = valid_sub_total / valid_total
                weight_total += outside_weight

                obs_dict = {
                    key: value
                    for key, value in batch.items()
                    if key in representation_input_buffer_specs
                }
                legal_actions = obs_dict["legal_actions"]

                v_target_list = []
                has_played_list = []
                v_trace_policy_target_list = []

                with torch.no_grad():
                    pi, _, log_pi, _ = self.net_offline.inference(obs_dict, valid)
                    pi_target, v_target, _, _ = self.net_target.inference(
                        obs_dict, valid
                    )
                    _, _, log_pi_prev, _ = self.net_reg.inference(obs_dict, valid)
                    _, _, log_pi_prev_, _ = self.net_reg_.inference(obs_dict, valid)

                policy_pprocessed = self.finetune.process_policy(
                    pi, legal_actions, self.n_discrete, self.epsilon_threshold
                )
                log_policy_reg = log_pi - (
                    alpha * log_pi_prev + (1 - alpha) * log_pi_prev_
                )

                for player in range(2):
                    reward = rewards[:, :, player]  # [T, B, Player]
                    v_target_, has_played, policy_target_ = v_trace(
                        v_target,
                        valid,
                        player_id,
                        policy,
                        policy_pprocessed,
                        log_policy_reg,
                        _player_others(player_id, valid, player),
                        action_oh,
                        reward,
                        player,
                        lambda_=1.0,
                        c=self.c_bar,
                        rho=self.rho_bar,
                        eta=self.eta,
                        gamma=self.gamma_vtrace,
                    )
                    v_target_list.append(v_target_)
                    has_played_list.append(has_played)
                    v_trace_policy_target_list.append(policy_target_)

                value_loss, neurd_loss, total_loss = self.net_offline.backpropagate(
                    obs_dict,
                    v_target_list,
                    v_trace_policy_target_list,
                    has_played_list,
                    valid,
                    player_id,
                    legal_actions,
                    clip=self.neurd_clip,
                    threshold=self.beta,
                    outside_weight=outside_weight,
                )
                # update logs per chunk
                if self.n % log_mod == 0 and self.wandb:
                    value_loss_ += value_loss #already multiplied by outside weight"
                    neurd_loss_ += neurd_loss
                    total_loss_ += total_loss
                    avg_traj_len_ += valid.float().sum(0).mean().item() / n_chunks
                    actor_learner_kld_ += (
                        torch.where(
                            valid.unsqueeze(-1).repeat(1, 1, 9),
                            torch.where(
                                legal_actions, pi * (torch.log(pi) - torch.log(policy)), 0
                            ),
                            torch.zeros_like(legal_actions),
                        )
                        .sum(-1).mean()
                        .item()
                        * outside_weight
                    )
                    log_legal = torch.log(
                        torch.nn.functional.normalize(legal_actions.to(torch.float), dim=-1, p=1)
                    )
                    entropy_ += (
                        torch.where(
                            valid.unsqueeze(-1).repeat(1, 1, 9),
                            torch.where(legal_actions, pi * (torch.log(pi) - log_legal), 0),
                            torch.zeros_like(legal_actions),
                        )
                    ).sum(-1).mean().item() * outside_weight

                    entropy_target_ += (
                        torch.where(
                            valid.unsqueeze(-1).repeat(1, 1, 9),
                            torch.where(legal_actions, pi_target * (torch.log(pi_target) - log_legal), 0),
                            torch.zeros_like(legal_actions),
                        )
                    ).sum(-1).mean().item() * outside_weight
            # end chunk loop

            assert((weight_total - 1)**2 < .001**2)

            if self.n % log_mod == 0 and self.wandb:

                to_log = {
                    "n": self.total_steps,
                    "loss_v": value_loss_,
                    "loss_nerd": neurd_loss_,
                    "loss_total": total_loss_,
                    "traj_len": avg_traj_len_,
                    "actor_learner_kld_": actor_learner_kld_,
                    "entropy_": entropy_,
                    "entropy_target_": entropy_target_,
                }
                wandb.log(to_log, step=self.total_steps)
                print(avg_traj_len_)

            nn.utils.clip_grad_norm_(self.net_offline.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.net_online.load_state_dict(self.net_offline.state_dict())
            with self.online_hash.get_lock():
                self.online_hash.value = randint(0, 2**20 - 1)
            self.net_target.step_weights(self.net_offline, lr=self.gamma_averaging)

            self.n += 1
            self.total_steps += 1

            if self.n == delta_m:
                self.net_reg_.load_state_dict(self.net_reg.state_dict())
                self.net_reg.load_state_dict(self.net_target.state_dict())
                self.m += 1
                self.n = 0

            first_step = False

            print(f"learn time: {int(time.time() - learn_start_time)} sec.", "\n")

    def _worker_loop(
        self,
        n_workers=2,
        n_players_per_worker=16,
        num_games=2**20,
        minutes_to_reset=120,
    ):

        self.net_online.share_memory()

        while True:

            print("Starting workers")
            for worker_index in range(n_workers):
                worker = SelfPlayWorker(
                    worker_index,
                    num_games,
                    n_players_per_worker,
                    team=None,
                    battle_format=self.format,
                    obs_dict=self.obs_dict,
                    shared_hash=self.online_hash,
                )
                process = self.ctx.Process(
                    target=worker.run,
                    args=(
                        self.net_online,
                        self.replay_buffer,
                    ),
                    name=repr(worker),  # TODO Doesn't have a name?
                )
                self.processes.append(process)
                self.workers.append(worker)

            for process in self.processes:
                process.start()

            ###
            time.sleep(minutes_to_reset * 60)
            ###
            with self.replay_buffer.lock:
                for process in self.processes:
                    process.terminate()
                print("Terminating workers")
                time.sleep(5)
                for process in self.processes:
                    process.join()
                print("Joined workers")
                self.workers = []
                self.processes = []

                self.replay_buffer.reset()
                print("Reset index_cache")

    def run(
        self,
        n_workers,
        n_players_per_worker,
        minutes_per_worker_reset=0.5,
        n_chunks=16,
        checkpoint_mod=10,
        log_mod=1,
    ):
        threads: List[threading.Thread] = []

        learn_thread = threading.Thread(
            target=self._learn,
            args=(n_chunks, checkpoint_mod, log_mod),
            name="Learning Thread",
        )
        threads.append(learn_thread)

        worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(n_workers, n_players_per_worker, 2**20, minutes_per_worker_reset),
            name="Worker Thread",
        )
        threads.append(worker_thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()


if __name__ == "__main__":
    batch_size = 2**11

    test = RNaD(
        format="gen3sampleteamrandbats",
        bounds=(4, 8, 16),
        delta_m=(100, 500, 1000),
        eta=0.2,
        lr=5 * 10**-4,
        gamma_averaging=0.01,
        batch_size=batch_size,
        buffer_retain=.9,
        epsilon_threshold=0.05,
        n_discrete=24,
        # directory_name=f"BasketNet4Test-{int(time.time())}",
        directory_name="cleffa",
        replay_buffer_size=batch_size * 1.2,
        net_init_params={
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
        wandb=True,
        # gamma_vtrace=0.95,
        # desc='lr=5e-4, gamma=.01 went bad at m=21 (1000 steps..). Logits are all one hots and values=-1.883. Using paper lr/gamma instead. Also increased n_disc from 16 to 24 and made delta_m start at 500'
    )

    test._initialize()

    n_workers = 2
    n_players_per_worker = 12
    minutes_per_worker_reset = 60 * 3

    # net/buffer checkpoint params are kwargs in RNaD._learn()

    test.run(
        n_workers,
        n_players_per_worker,
        minutes_per_worker_reset,
        n_chunks=16,
        checkpoint_mod=10,
        log_mod=1,
    )
