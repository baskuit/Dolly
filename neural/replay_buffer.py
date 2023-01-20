import torch
from os import path, mkdir, remove, scandir
from time import sleep, time
from typing import Dict, List

from neural.buffer_specs import replay_buffer_specs, MAX_T

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

Buffers = Dict[str, List[torch.Tensor]]


class ReplayBuffer:
    def __init__(
        self,
        ctx,
        directory,
        device=torch.device("cpu"),
        sub_batch_size=64,
        max_sub_batch=16,
    ):
        self.ctx = ctx
        self.device = device

        self.sub_batches: List[int] = []
        self.sub_batch_size = sub_batch_size
        # This is intended to fit in learner memory, so 2**6/7 i guess
        self.max_sub_batch = max_sub_batch
        self.directory = path.join(directory, "buffer")

        self.buffer_size = int(1.5 * sub_batch_size)

        self.index_cache = self.ctx.Manager().dict()
        self.done_cache = [self.ctx.Value("i", 0) for i in range(self.buffer_size)]
        self.turn_counters = [self.ctx.Value("i", 0) for i in range(self.buffer_size)]
        self.full_queue = self.ctx.Queue(maxsize=self.buffer_size)
        self.free_queue = self.ctx.Queue(maxsize=self.buffer_size)

        for m in range(self.buffer_size):
            self.free_queue.put(m)

        self.lock = self.ctx.Lock()
        self.buffers: Buffers = None

        self._initialize()

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

        self.buffers = ReplayBuffer._create_buffers(
            replay_buffer_specs, self.buffer_size
        )

    def get_sub_batches(self, batch_size: int):

        num_sub_batches = batch_size // self.sub_batch_size
        assert num_sub_batches * self.sub_batch_size == batch_size

        while len(self.sub_batches) < num_sub_batches:
            sleep(1)

        self.sub_batches = sorted(self.sub_batches, reverse=True)
        sub_batches = self.sub_batches[:num_sub_batches]

        for sub_batch_timestamp in sub_batches:
            sub_batch_path = path.join(self.directory, sub_batch_timestamp)
            sub_batch = torch.load(sub_batch_path, map_location=self.device)
            yield sub_batch

        self._pop_sub_batches()

    def _pop_sub_batches(self):
        self.sub_batches = sorted(self.sub_batches, reverse=True)
        for sub_batch_timestamp in self.sub_batches[self.max_sub_batch :]:
            sub_batch_path = path.join(self.directory, sub_batch_timestamp)
            remove(sub_batch_path)
        self.sub_batches = self.sub_batches[: self.max_sub_batch]

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
                pass
            else:
                for key, value in step.items():
                    self.buffers[key][index][turn][...] = value
                self.buffers["valid"][index][turn][...] = 1
                self.buffers["rewards"][index][turn][...] = 0
            self.turn_counters[index].value += 1

    def clean(self, battle_tag):
        index = self._get_index(battle_tag)
        with self.done_cache[index].get_lock():
            self.done_cache[index].value = 0
            self.index_cache.pop(battle_tag, None)
            self._reset_index(index)
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
                    self._reset_index(index)
                self.done_cache[index].value = 0
                self.index_cache.pop(battle_tag, None)

    def _reset_index(self, index: int):
        with self.turn_counters[index].get_lock():
            self.turn_counters[index].value = 0
        self.buffers["valid"][index][...] = 0
        self.buffers["legal_actions"][index][...] = 1

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

    def _save_live_buffer(
        self,
    ):

        sleep(1)  # give worker thread time to acquire

        with self.lock:

            indices = [self.full_queue.get() for _ in range(self.max_sub_batch)]

            valids = torch.stack([self.buffers["valid"][m] for m in indices])
            lengths = valids.sum(-1)

            indices = list(
                zip(*sorted(list(zip(lengths.tolist(), indices)), key=lambda x: x[0]))
            )[1]

            batch = {
                key: torch.stack(
                    [self.buffers[key][index] for index in indices],
                    dim=1,
                )[: lengths.max().item()]
                for key in self.buffers
            }

            for index in indices:
                self._reset_index(index)
                self.free_queue.put(index)

            sub_batch_timestamp = str(int(time() * 1000))
            file_name = path.join(self.directory, sub_batch_timestamp)
            self.sub_batches.append(sub_batch_timestamp)
            torch.save(batch, file_name)

    def save_loop(
        self,
    ):

        while True:
            self._save_live_buffer()
