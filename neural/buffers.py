import torch
import random
import logging
import time


from typing import Dict, List

from neural.buffer_specs import replay_buffer_specs, MAX_T

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

Buffers = Dict[str, List[torch.Tensor]]


class ReplayBuffer:
    def __init__(
        self,
        ctx,
        num_buffers: int = 512,
        specs: Buffers = replay_buffer_specs,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        retain=0,
    ):
        self.ctx = ctx
        self.num_buffers = num_buffers
        self.specs = specs
        self.device = device
        self.retain = retain

        self.index_cache = self.ctx.Manager().dict()
        self.done_cache = [self.ctx.Value("i", 0) for i in range(num_buffers)]

        self.buffers = ReplayBuffer._create_buffers(specs, num_buffers)

        self.turn_counters = [self.ctx.Value("i", 0) for i in range(num_buffers)]

        self.full_queue = self.ctx.Queue(maxsize=self.num_buffers)
        self.free_queue = self.ctx.Queue(maxsize=self.num_buffers)

        for m in range(num_buffers):
            self.free_queue.put(m)

        self.lock = self.ctx.Lock()

    def reset(self):
        self.index_cache = self.ctx.Manager().dict()
        self.done_cache = [self.ctx.Value("i", 0) for i in range(self.num_buffers)]
        self.turn_counters = [self.ctx.Value("i", 0) for i in range(self.num_buffers)]
        self.free_queue = self.ctx.Queue(maxsize=self.num_buffers)
        self.full_queue = self.ctx.Queue(maxsize=self.num_buffers)
        for index in range(self.num_buffers):
            self.free_queue.put(index)
            self.buffers["valid"][index][...] = 0
            self.buffers["legal_actions"][index][...] = 1

    @staticmethod
    def _create_buffers(specs: Buffers, num_buffers: int) -> Buffers:
        buffers: Buffers = {key: [] for key in specs}
        for _ in range(num_buffers):
            for key in buffers:
                if key == "legal_actions":
                    buffers[key].append(torch.ones(**specs[key]).share_memory_())
                else:
                    buffers[key].append(torch.zeros(**specs[key]).share_memory_())
        return buffers

    def _get_index(self, battle_tag: str) -> int:
        if battle_tag in self.index_cache:
            index = self.index_cache[battle_tag]

        else:
            index = self.free_queue.get()
            self.index_cache[battle_tag] = index

        return index

    def store_sample(self, battle_tag: str, step: Dict[str, torch.Tensor]):
        index = self._get_index(battle_tag)
        with self.turn_counters[index].get_lock():
            turn = self.turn_counters[index].value
            if turn >= MAX_T:
                print("store_sample skipped. too long")
            else:
                for key, value in step.items():
                    self.buffers[key][index][turn][...] = value
                self.buffers["valid"][index][turn][...] = 1
                self.buffers["rewards"][index][turn][...] = 0
                self.buffers["entry_time"][index][...] = int(time.time())
            self.turn_counters[index].value += 1

    def clean(self, battle_tag):
        index = self._get_index(battle_tag)
        with self.done_cache[index].get_lock():
            self.done_cache[index].value = 0
            self.index_cache.pop(battle_tag, None)
            self.reset_index(index)
        self.free_queue.put(index)

    def ammend_reward(self, battle_tag: str, pid: int, reward: int):
        index = self._get_index(battle_tag)
        with self.turn_counters[index].get_lock():
            turn = self.turn_counters[index].value - 1
            if turn < MAX_T:
                self.buffers["rewards"][index][turn, pid][...] = reward
            else:
                print("store_sample skipped. too long")
        return

    def register_done(self, battle_tag: str):
        index = self._get_index(battle_tag)
        with self.done_cache[index].get_lock():
            self.done_cache[index].value += 1
            dones = self.done_cache[index].value
            if dones >= 2:
                with self.turn_counters[index].get_lock():
                    turn = self.turn_counters[index].value
                if turn <= MAX_T:
                    self.full_queue.put(index)
                else:
                    self.free_queue.put(index)
                    self.reset_index(index)
                self.done_cache[index].value = 0
                self.index_cache.pop(battle_tag, None)
        return

    def reset_index(self, index: int):
        with self.turn_counters[index].get_lock():
            self.turn_counters[index].value = 0
        # for key in self.buffers.keys():
        #     self.buffers[key][index][...] = 0
        self.buffers["valid"][index][...] = 0
        self.buffers["legal_actions"][index][...] = 1
        self.buffers["entry_time"][index][...] = 0

    def get_batches(self, batch_size: int, n_chunks=2**3):

        time.sleep(1)  # give worker thread time to acquire

        with self.lock:
            indices = [self.full_queue.get() for _ in range(batch_size)]

            valids = torch.stack([self.buffers["valid"][m] for m in indices])
            lengths = valids.sum(-1)
            max_index = lengths.max().item()
            entry_times = [self.buffers["entry_time"][m] for m in indices]

            indices = list(
                zip(*sorted(list(zip(lengths.tolist(), indices)), key=lambda x: x[0]))
            )[1]
            indices_by_time = list(
                zip(*sorted(list(zip(entry_times, indices)), key=lambda x: x[0]))
            )[1]

            batch = {
                key: torch.stack(
                    [self.buffers[key][index] for index in indices],
                    dim=1,
                )[:max_index]
                for key in self.buffers
            }
            retain_idx = min(batch_size, max(0, int(batch_size * self.retain)))
            for index in indices_by_time[retain_idx:]:
                self.reset_index(index)
                self.free_queue.put(index)
            for index in indices_by_time[:retain_idx]:
                self.full_queue.put(index)
            batch_slice = batch_size // n_chunks
            batches = [
                {
                    k: t[:, chunk * batch_slice : (chunk + 1) * batch_slice].to(
                        device=self.device, non_blocking=True
                    )
                    for k, t in batch.items()
                }
                for chunk in range(n_chunks)
            ]
            return batches
