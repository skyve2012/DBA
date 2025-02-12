import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch
        

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        # Use weighted sampling if weights are provided
        if weights is not None:
            sampler = WeightedRandomSampler(weights, replacement=True, num_samples=len(dataset), generator=torch.Generator())
        else:
            # Use random sampling without replacement by default
            sampler = RandomSampler(dataset, replacement=False)

        # Set up the DataLoader
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True
        )

    def __iter__(self):
        # Return the iterator for the DataLoader
        return iter(self.data_loader)

    def __len__(self):
        # Return the number of batches available in the DataLoader
        return len(self.data_loader)

class FastDataLoader:
    """
    DataLoader wrapper with slightly improved speed by not respawning worker processes at every epoch.
    Designed to iterate over the dataset exactly once per epoch.
    """
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()
        
        self.batch_sampler = BatchSampler(
            RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        # Create a DataLoader with the batch sampler
        self.data_loader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=self.batch_sampler
        )
        self.dataset = dataset

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.batch_sampler)