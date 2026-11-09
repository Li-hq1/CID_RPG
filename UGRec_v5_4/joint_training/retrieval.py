"""Optional CPU HNSW serving index, built once for a frozen AE snapshot."""

import numpy as np
import torch
from torch.nn import functional as F


class CatalogIndex:
    def __init__(self, items, metric="l2", neighbors=32, ef_search=128):
        try:
            import faiss
        except ImportError as error:
            raise ImportError("HNSW retrieval requires the optional faiss-cpu package.") from error
        faiss.omp_set_num_threads(2)
        self.metric, self.items = metric, items.detach()
        values = self._vectors(items)
        faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        self.index = faiss.IndexHNSWFlat(items.shape[1], neighbors, faiss_metric)
        self.index.hnsw.efConstruction = 160
        self.index.hnsw.efSearch = ef_search
        self.index.add(values)

    def _vectors(self, values):
        if self.metric == "cosine":
            values = F.normalize(values.float(), dim=-1)
        return np.ascontiguousarray(values.detach().float().cpu().numpy())

    @torch.no_grad()
    def search(self, queries, depth, metric):
        if metric != self.metric:
            raise ValueError("Index metric differs from requested retrieval metric.")
        _, ids = self.index.search(self._vectors(queries), min(depth, len(self.items)))
        ids = torch.from_numpy(ids).to(device=queries.device, dtype=torch.long)
        if (ids < 0).any():
            raise RuntimeError("HNSW returned an unfilled candidate slot.")
        # Recompute exact distances only for returned candidates, preserving
        # support aggregation semantics while avoiding a full-catalog GEMM.
        keys = self.items[ids]
        if metric == "l2":
            distances = (queries[:, None].float() - keys.float()).square().sum(-1)
        elif metric == "cosine":
            distances = 1 - (F.normalize(queries[:, None].float(), dim=-1)
                             * F.normalize(keys.float(), dim=-1)).sum(-1)
        else:
            distances = -(queries[:, None].float() * keys.float()).sum(-1)
        # Stable ID tie-break, then distance; HNSW candidate discovery is approximate.
        order = ids.argsort(dim=1, stable=True)
        ids, distances = ids.gather(1, order), distances.gather(1, order)
        order = distances.argsort(dim=1, stable=True)
        return ids.gather(1, order) + 1, distances.gather(1, order)
