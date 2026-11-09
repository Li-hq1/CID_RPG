"""Content-only candidate sampling from the compression training population."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class LocalCandidates:
    def __init__(self, embeddings, train_indices, query_indices, config):
        self.train = np.asarray(train_indices, dtype=np.int64)
        self.config = config
        if len(self.train) < 2 or len(np.unique(self.train)) != len(self.train):
            raise ValueError("Local relations need at least two unique training items.")
        queries = np.unique(query_indices)
        search = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
        search.fit(embeddings[self.train])
        count = min(len(self.train), config.local_middle_pool + 1)
        self.neighbors = {}
        # Chunk queries so the exact search never materializes an all-item square matrix.
        for start in range(0, len(queries), config.eval_batch_size):
            batch = queries[start:start + config.eval_batch_size]
            positions = search.kneighbors(embeddings[batch], n_neighbors=count, return_distance=False)
            for anchor, row in zip(batch, positions):
                items = self.train[row]
                self.neighbors[int(anchor)] = items[items != anchor][:config.local_middle_pool]
        self.population = set(self.train.tolist())

    def sample(self, anchors, rng):
        config = self.config
        requested = config.local_near_count + config.local_middle_count + config.local_random_count
        width = min(requested, len(self.train) - int(any(int(a) in self.population for a in anchors)))
        result = []
        for anchor in anchors:
            neighbors = self.neighbors[int(anchor)]
            near = neighbors[:config.local_near_pool]
            middle = neighbors[config.local_near_pool:config.local_middle_pool]
            selected = np.concatenate((rng.choice(near, min(len(near), config.local_near_count), replace=False),
                                       rng.choice(middle, min(len(middle), config.local_middle_count), replace=False)))[:width]
            available = self.train[~np.isin(self.train, np.append(selected, anchor))]
            # Small populations fill missing near/middle slots with remaining training items.
            random = rng.choice(available, width - len(selected), replace=False)
            result.append(np.concatenate((selected, random)))
        return np.asarray(result, dtype=np.int64)


def latent_statistics(raw):
    """Detached, epoch-level statistics shared by anchors and all candidates."""
    values = raw.astype(np.float64)
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), 1e-6).astype(np.float32)
