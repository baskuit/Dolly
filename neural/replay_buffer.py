import torch
from os import path, mkdir, rmdir, scandir
from time import sleep
from typing import List

class ReplayBuffer:
    def __init__(
        self,
        ctx,
        device=torch.device('cpu'),
        sub_batch_size=64,
        max_sub_batch=16,
    ):
        self.ctx = ctx
        self.device = device
        max_sub_batch=16,

        self.sub_batches : List[int] = []
        self.sub_batch_size = sub_batch_size
        # This is intended to fit in learner memory, so 2**6/7 i guess
        self.max_sub_batch = max_sub_batch
        self.directory = path.join(
            path.dirname(path.realpath(__file__)), "buffer"
        )
        buffer_size = int(1.2 * sub_batch_size)
        self.index_cache = self.ctx.Manager().dict()
        self.done_cache = [self.ctx.Value("i", 0) for i in range(buffer_size)]
        self.turn_counters = [self.ctx.Value("i", 0) for i in range(buffer_size)]
        self.full_queue = self.ctx.Queue(maxsize=self.num_buffers)
        self.free_queue = self.ctx.Queue(maxsize=self.num_buffers)

        for m in range(buffer_size):
            self.free_queue.put(m)

        self.lock = self.ctx.Lock()

    @staticmethod
    def _create_buffers(specs, num_buffers: int):
        buffers = {key: [] for key in specs}
        for _ in range(num_buffers):
            for key in buffers:
                if key == "legal_actions":
                    buffers[key].append(torch.ones(**specs[key]).share_memory_())
                else:
                    buffers[key].append(torch.zeros(**specs[key]).share_memory_())
        return buffers

    def _initialize(self):

        if not path.exists(self.directory):
            mkdir(self.directory)

        self.sub_batches = [
            int(path.relpath(f.path, self.directory))
            for f in scandir(self.directory)
            if f.is_dir()
        ]

        if not self.sub_batches:
            pass
        else:
            pass

        self.buffer = self._create_buffers(self.sub_batch_size)

    def get_batch (self, batch_size: int):
        
        num_sub_batches = batch_size // self.sub_batch_size
        assert(num_sub_batches * self.sub_batch_size == batch_size)

        while len(self.sub_batches) < num_sub_batches:
            sleep(1)

        self.sub_batches = sorted(self.sub_batches, reverse=True)
        sub_batches = self.sub_batches[:num_sub_batches]

        for sub_batch_timestamp in sub_batches:
            sub_batch_path = path.join(self.directory, sub_batch_timestamp)
            sub_batch = torch.load(sub_batch_path, map_location=self.device)
            yield sub_batch

            valids = torch.stack([self.buffers["valid"][m] for m in indices])
            lengths = valids.sum(-1)
            max_index = lengths.max().item()
            entry_times = [self.buffers["entry_time"][m] for m in indices]

            indices = list(
                zip(*sorted(list(zip(lengths.tolist(), indices)), key=lambda x: x[0]))
            )[1]
            indices_by_time = list(
                zip(
                    *sorted(
                        list(zip(entry_times, indices)),
                        key=lambda x: x[0],
                        reverse=True,
                    )
                )
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


        self._pop_sub_batches()

    def _pop_sub_batches(self):

        self.sub_batches = sorted(self.sub_batches, reverse=True)
        for sub_batch_timestamp in self.sub_batches[self.max_sub_batch:]:
            sub_batch_path = path.join(self.directory, sub_batch_timestamp)
            rmdir(sub_batch_path)
        self.sub_batches = self.sub_batches[:self.max_sub_batch]


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