"""Bounded CPU replay of high-loss aligned samples, included in checkpoints."""
import torch


class HardSampleReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.source, self.destination = [], []

    def add(self, source, destination, source_losses, destination_losses):
        for entries, batch, losses in ((self.source, source, source_losses),
                                       (self.destination, destination, destination_losses)):
            for index, loss in enumerate(losses.detach().cpu().tolist()):
                if not torch.isfinite(torch.tensor(loss)):
                    continue
                sample = tuple(torch.as_tensor(value[index]).detach().cpu().float().clone() for value in batch)
                entries.append((float(loss), sample))
            entries.sort(key=lambda item: item[0], reverse=True)
            del entries[self.capacity:]

    def batch(self, count):
        if len(self.source) < count or len(self.destination) < count:
            return None
        def stack(entries):
            selected = entries[:count]
            del entries[:count]
            return tuple(torch.stack([sample[position] for _, sample in selected]) for position in range(4))
        return stack(self.source), stack(self.destination)

    def state_dict(self):
        return dict(source=self.source, destination=self.destination)

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {'source', 'destination'}:
            raise ValueError('Invalid hard-sample replay state')
        for name in ('source', 'destination'):
            entries = state[name]
            if not isinstance(entries, list) or len(entries) > self.capacity:
                raise ValueError('Invalid hard-sample replay capacity')
            for loss, sample in entries:
                if not isinstance(loss, (int, float)) or not torch.isfinite(torch.tensor(loss)) or len(sample) != 4:
                    raise ValueError('Invalid hard-sample replay entry')
                if any(not isinstance(value, torch.Tensor) or value.ndim != 3 or not torch.isfinite(value).all() for value in sample):
                    raise ValueError('Invalid hard-sample replay tensor')
            setattr(self, name, entries)
