import re
import warnings
import random
from torch.utils.data import BatchSampler

def find_nearest_bucket(h, w, bucket_options):
    """Finds the closes bucket to the given height and width."""
    min_metric = float("inf")
    best_bucket_idx = None
    for bucket_idx, (bucket_h, bucket_w) in enumerate(bucket_options):
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket_idx = bucket_idx
    return best_bucket_idx


def parse_buckets_string(buckets_str):
    """Parses a string defining buckets into a list of (height, width) tuples."""
    if not buckets_str:
        raise ValueError("Bucket string cannot be empty.")

    bucket_pairs = buckets_str.strip().split(";")
    parsed_buckets = []
    for pair_str in bucket_pairs:
        match = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", pair_str)
        if not match:
            raise ValueError(f"Invalid bucket format: '{pair_str}'. Expected 'height,width'.")
        try:
            height = int(match.group(1))
            width = int(match.group(2))
            if height <= 0 or width <= 0:
                raise ValueError("Bucket dimensions must be positive integers.")
            if height % 8 != 0 or width % 8 != 0:
                warnings.warn(f"Bucket dimension ({height},{width}) not divisible by 8. This might cause issues.")
            parsed_buckets.append((height, width))
        except ValueError as e:
            raise ValueError(f"Invalid integer in bucket pair '{pair_str}': {e}") from e

    if not parsed_buckets:
        raise ValueError("No valid buckets found in the provided string.")

    return parsed_buckets


class BucketBatchSampler(BatchSampler):
    """
    基于图像大小将数据集索引分组到桶中的批次采样器
    """
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size应为正整数，获取到的值为{batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last应为布尔值，获取到的值为{drop_last}")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        self.bucket_indices = {}
        for idx, bucket_idx in enumerate(dataset.bucket_indices):
            if bucket_idx not in self.bucket_indices:
                self.bucket_indices[bucket_idx] = []
            self.bucket_indices[bucket_idx].append(idx)
        
        self.batches = []
        self.sampler_len = 0
        
        for bucket_idx, indices in self.bucket_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                self.batches.append(batch)
                self.sampler_len += 1
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return self.sampler_len
