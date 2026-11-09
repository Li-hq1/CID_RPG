# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os
import tempfile
import numpy as np
from logging import getLogger
from misc.logging import log
from data.embeddings import (
    embedding_settings, encode_sentences, read_sentence_embeddings,
    sentence_embedding_cache_path,
)


class TigerDiffTokenizer:
    def __init__(self, config: dict, dataset):
        self.config = config
        self.logger = getLogger()
        self.eos_token = None
        self.collate_fn = {'train': None, 'val': None, 'test': None}
        self.padding_token = 0
        self.cache_dir = dataset.cache_dir
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping["id2item"]
        self.sent_embs = self._init_tokenizer(dataset)
        self.item2tokens = self.sent_embs
        self.REGENERATE_SAMPLE = False
        self.ignored_label = -100

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns:
            int: The maximum token sequence length.
        """
        return self.config["max_item_seq_len"]

    def _encode_sent_emb(self, dataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset: The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config["metadata"] == "sentence", (
            "TIGERTokenizer only supports sentence metadata."
        )

        meta_sentences = []  # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping["id2item"][i]])

        sent_embs = encode_sentences(meta_sentences, self.config)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Interrupted extraction must never leave a cache that looks complete.
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(output_path) or ".", delete=False) as f:
            temporary_path = f.name
        try:
            sent_embs.tofile(temporary_path)
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return sent_embs

    def _load_sent_emb(self, dataset):
        path = sentence_embedding_cache_path(self.config, self.cache_dir)
        if os.path.exists(path):
            self.log(f"[TOKENIZER] Loading sentence embeddings from {path}...")
            return read_sentence_embeddings(path, self.config["sent_emb_dim"], dataset.n_items - 1)
        self.log(f"[TOKENIZER] Encoding sentence embeddings with {self.config['sent_emb_model']}...")
        return self._encode_sent_emb(dataset, path)

    def _get_items_for_training(self, dataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset: The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        items_for_training = set()
        for item_seq in dataset.split_data["train"]["item_seq"]:
            for item in item_seq:
                items_for_training.add(item)
        self.log(
            f"[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}"
        )
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _fit_transform_sentence_embeddings(
        self, sent_embs: np.ndarray, train_mask: np.ndarray
    ) -> np.ndarray:
        if train_mask.shape[0] != sent_embs.shape[0]:
            raise ValueError(
                "train_mask must have one entry for each non-padding item embedding."
            )
        if not np.any(train_mask):
            raise ValueError(
                "Cannot fit sentence embedding statistics because no training items were found."
            )

        n_components = int(self.config.get("sent_emb_pca", 0))
        if n_components < 0:
            raise ValueError("sent_emb_pca must be non-negative.")
        train_sent_embs = sent_embs[train_mask]
        if n_components > 0:
            from sklearn.decomposition import PCA

            max_components = min(train_sent_embs.shape[0], train_sent_embs.shape[1])
            if n_components > max_components:
                raise ValueError(
                    f"sent_emb_pca={n_components} exceeds the maximum PCA components "
                    f"available from training items ({max_components})."
                )

            pca = PCA(n_components=n_components, whiten=True)
            pca.fit(train_sent_embs)
            sent_embs = pca.transform(sent_embs)

        train_sent_embs = sent_embs[train_mask]
        mean = np.mean(train_sent_embs, axis=0)
        std = np.std(train_sent_embs, axis=0)
        std = np.where(std < 1e-6, 1e-6, std)
        sent_embs = (sent_embs - mean) / std
        return sent_embs.astype(np.float32)

    def _tokenized_cache_path(self) -> str:
        method = self.config.get("sent_emb_compression", "pca")
        cache_config = {
            "cache_version": 1,
            "tokenizer": self.__class__.__name__,
            "metadata": self.config.get("metadata"),
            "sent_emb_model": self.config.get("sent_emb_model"),
            "sent_emb_dim": self.config.get("sent_emb_dim"),
            "sent_emb_pca": self.config.get("sent_emb_pca"),
            "max_item_seq_len": self.config.get("max_item_seq_len"),
            "label_policy": "history_window_last_next_item",
        }
        if any(key in self.config for key in (
            "sent_emb_max_length", "sent_emb_dtype", "sent_emb_prompt", "sent_emb_normalize",
        )):
            cache_config["embedding_settings"] = embedding_settings(self.config)
        cache_key = hashlib.md5(
            json.dumps(cache_config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        return os.path.join(
            self.cache_dir, "processed", f"tokenized_datasets_TigerDiff_{cache_key}"
        )

    def _init_tokenizer(self, dataset):
        """
        Returns:
            numpy.ndarray: The sentence embeddings.
        """

        method = self.config.get("sent_emb_compression", "pca")

        sent_embs = self._load_sent_emb(dataset)

        # PCA/statistics are fit on training items only to avoid val/test leakage.
        if int(self.config.get("sent_emb_pca", 0)) > 0:
            self.log("[TOKENIZER] Applying PCA to sentence embeddings...")
        else:
            self.log("[TOKENIZER] Skipping PCA for sentence embeddings...")

        train_mask = self._get_items_for_training(dataset)
        sent_embs = self._fit_transform_sentence_embeddings(sent_embs, train_mask)

        self.log(f"[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}")

        return sent_embs

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        """
        Tokenizes the first n items in the given item_seq.
        The losses for the first n items can be computed by only forwarding once.

        Args:
            item_seq (list): The item sequence that contains the first n items.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, labels, and seq_lens.
        """
        input_ids = [
            self.item2id[item] for item in item_seq[:-1]
        ]  # 这里item_seq[:-1]是因为最后一个item是target item
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2id[item] for item in item_seq[1:]]  # target
        labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        """
        Tokenizes the later items in the item sequence.
        Only the last one items are used as the target item.

        Args:
            item_seq (list): The item sequence.

        Returns:
            tuple: A tuple containing the tokenized input IDs, attention mask, labels, and seq_lens.
        """
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.

        Args:
            example (dict): The input example containing the item sequence.
            split (str): The split type ('train' or 'val' or 'test').

        Returns:
            dict: A dictionary containing the tokenized input, attention mask, and labels.
        """
        max_item_seq_len = self.config["max_item_seq_len"]
        item_seq = example["item_seq"][0]
        if split == "train":
            # 这里n_return_examples是51，因为max_item_seq_len=50，最后一个item作为target item
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

            # Tokenize the first n items if len(item_seq) <= max_item_seq_len + 1
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                # Add 1 as the target item is not included in the input sequence
                item_seq=item_seq[: min(len(item_seq), max_item_seq_len + 1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = (
                [input_ids],
                [attention_mask],
                [labels],
                [seq_lens],
            )

            # Tokenize the later items if len(item_seq) > max_item_seq_len + 1
            # 以51为窗口，滑动生成训练样本，只有最后一个item作为target item
            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i : i + max_item_seq_len + 1]  # 51个item
                input_ids, attention_mask, labels, seq_lens = (
                    self._tokenize_later_items(cur_item_seq)
                )
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "labels": all_labels,
                "seq_lens": all_seq_lens,
            }
        else:
            # 从后往前划定窗口51个，且最后一个作为target
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len + 1) :], pad_labels=False
            )
            return {
                "input_ids": [input_ids],
                "attention_mask": [attention_mask],
                "labels": [
                    labels[-1:]
                ],  # 由于全部取最后一个label，这里没有对label padding，而是取了最后一个label值
                "seq_lens": [seq_lens],
            }

    def tokenize(self, datasets: dict) -> dict:
        """
        新增了保存sample，避免每次都重新生成数据样本，
        如果修改了数据生成逻辑记得重新生成
        """
        # 检查是否存在缓存的tokenized数据
        tokenized_cache_path = self._tokenized_cache_path()

        if os.path.exists(tokenized_cache_path) and not self.REGENERATE_SAMPLE:
            self.log(
                f"[TOKENIZER] Loading tokenized datasets from {tokenized_cache_path}..."
            )
            try:
                from datasets import load_from_disk

                tokenized_datasets = {}
                for split in datasets:
                    split_path = os.path.join(tokenized_cache_path, split)
                    if os.path.exists(split_path):
                        tokenized_datasets[split] = load_from_disk(split_path)
                        tokenized_datasets[split].set_format(type="torch")
                    else:
                        self.log(
                            f"[TOKENIZER] Cache for {split} not found, tokenizing..."
                        )
                        tokenized_datasets[split] = self._tokenize_split(
                            datasets[split], split
                        )
                return tokenized_datasets
            except Exception as e:
                self.log(
                    f"[TOKENIZER] Failed to load cached tokenized datasets: {e}, tokenizing from scratch..."
                )

        # 如果没有缓存，进行tokenization
        self.log("[TOKENIZER] Tokenizing datasets...")
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = self._tokenize_split(datasets[split], split)

        # 保存tokenized数据到缓存
        os.makedirs(tokenized_cache_path, exist_ok=True)
        for split in datasets:
            split_path = os.path.join(tokenized_cache_path, split)
            tokenized_datasets[split].save_to_disk(split_path)
        self.log(f"[TOKENIZER] Tokenized datasets saved to {tokenized_cache_path}")

        return tokenized_datasets

    def _tokenize_split(self, dataset, split: str):
        """
        Tokenizes a single dataset split.

        Args:
            dataset: The dataset to tokenize
            split (str): The split name

        Returns:
            The tokenized dataset
        """
        tokenized_dataset = dataset.map(
            lambda t: self.tokenize_function(t, split),
            batched=True,  # 批处理但bs=1，批处理的格式
            batch_size=1,
            remove_columns=dataset.column_names,  # 删除原始数据中的列，只保留tokenized后的新列
            # datasets 4.x creates a worker even for num_proc=1.
            num_proc=self.config["num_proc"] if self.config["num_proc"] > 1 else None,
            desc=f"Tokenizing {split} set: ",
        )
        tokenized_dataset.set_format(type="torch")
        return tokenized_dataset

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
