from pathlib import Path
from types import SimpleNamespace
import numpy as np
from config import CONFIG
from misc.artifacts import training_item_mask
from data.embeddings import sentence_embedding_cache_path

def fixture(directory):
    ids = ["[PAD]"] + [f"item{i}" for i in range(1, 13)]
    splits = {"train": {"item_seq": [ids[1:11]]}}
    dataset = SimpleNamespace(
        cache_dir=str(Path(directory) / "AmazonReviews2014" / "Beauty"),
        n_items=len(ids), id_mapping={"id2item": ids},
        item2id={item: i for i, item in enumerate(ids)}, user2id={"u": 1},
        split=lambda: splits,
    )
    dataset.split_data = splits
    config = {**CONFIG, "category": "Beauty", "dataset": "AmazonReviews2014", "device": "cpu",
              "geometry_ae_output_root": None, "sent_emb_compression": "pca", "sent_emb_dim": 6, "num_proc": 1, "max_item_seq_len": 4}
    embeddings = np.random.default_rng(42).normal(size=(12, 6)).astype(np.float32)
    path = Path(sentence_embedding_cache_path(config, dataset.cache_dir))
    path.parent.mkdir(parents=True)
    embeddings.tofile(path)
    return dataset, config, embeddings, training_item_mask(dataset)
