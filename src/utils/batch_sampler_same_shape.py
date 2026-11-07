from torch.utils.data import Sampler
from math import ceil
import random

class BatchSamplerSameShape(Sampler):
    r"""Yield a mini-batch of indices. The sampler will drop the last batch of
            an image size bin if it is not equal to ``batch_size``

    Args:
        examples (dict): List from dataset class.
        batch_size (int): Size of mini-batch.
    """

    """
        In this setup, we usually have the following data organization:
            - video: shape: (1, T, C, X, Y) # C=1 or C=2
            - ggf. warped_noise_sample (1, T, C, X, Y) # C=1 or C=2
            - ggf. model_concat_input: shape (1, T, 2, X, Y)
            - added_time_ids: shape: (4, )
            - fps: shape (1,)
            - attr, where we here actually only need the warping error. In any case
            # -> can be stacked if T, X, Y are the same across samples in a batch
    """

    def __init__(self, dataset, batch_size, indices=None,
        shuffle=False, group_shape_by='video', target_key_name : str = "video", assume_fixed_shape: bool = False
        ):
        self.batch_size = batch_size
        self.data = {}
        self.shuffle = shuffle
        self.indices = range(len(dataset)) if indices is None else indices
        self.assume_fixed_shape = assume_fixed_shape

        if self.assume_fixed_shape:
            self.data["fixed_shape"] = list(self.indices)
            self.total = ceil(len(self.data["fixed_shape"]) / self.batch_size)
            return

        for idx in self.indices:
            
            item = dataset[idx]

            video_shape = item[target_key_name].shape

            if group_shape_by == 'video':
                shape = video_shape
            else:
                raise NotImplementedError(f'group_shape_by = \'{group_shape_by}\' not implemented')

            if shape in self.data:
                self.data[shape].append(idx)
            else:
                self.data[shape] = [idx]

        self.total = 0
        for shape, indices in self.data.items():
            self.total += ceil(len(indices) / self.batch_size)
            
    def __iter__(self):
        batches = []

        for _, indices in self.data.items():
            if self.shuffle:
                random.shuffle(indices)                
            batch = []
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []
            if batch:
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.total