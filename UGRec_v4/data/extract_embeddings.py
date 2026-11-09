"""Extract item embeddings without starting recommendation training."""

import logging
import tempfile
from types import SimpleNamespace

from accelerate import Accelerator
import numpy as np

from data.amazon_reviews import AmazonReviews2014
from data.embeddings import sentence_embedding_cache_path
from data.tigerdiff_data import TigerDiffTokenizer
from misc.config import parse_config
from misc.seed import init_seed


def main():
    config, args = parse_config()
    accelerator = Accelerator()
    config.update(accelerator=accelerator, device=accelerator.device)
    logging.basicConfig(level=logging.INFO)
    init_seed(config["rand_seed"], config["reproducibility"])
    dataset = AmazonReviews2014(config)
    tokenizer = TigerDiffTokenizer.__new__(TigerDiffTokenizer)
    tokenizer.config = config
    tokenizer.cache_dir = dataset.cache_dir
    tokenizer.logger = logging.getLogger()
    with accelerator.main_process_first():
        if args.smoke_test:
            # Smoke output cannot be mistaken for a full category cache.
            sample = SimpleNamespace(
                n_items=min(5, dataset.n_items),
                item2meta=dataset.item2meta,
                id_mapping=dataset.id_mapping,
            )
            with tempfile.TemporaryDirectory() as directory:
                embeddings = tokenizer._encode_sent_emb(sample, f"{directory}/smoke.sent_emb")
            accelerator.print(f"Smoke extraction: shape={embeddings.shape}, dtype={embeddings.dtype}, "
                              f"norms={np.linalg.norm(embeddings, axis=1)}")
        else:
            embeddings = tokenizer._load_sent_emb(dataset)
            accelerator.print(f"Extracted/reused {embeddings.shape} float32 embeddings: "
                              f"{sentence_embedding_cache_path(config, dataset.cache_dir)}")


if __name__ == "__main__":
    main()
