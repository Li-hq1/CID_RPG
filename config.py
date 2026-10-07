"""Edit experiment defaults here. CLI overrides are declared in misc/config.py."""

CONFIG = {
    # ---------------------- data and cache ----------------------
    "category": "Sports_and_Outdoors",
    "cache_dir": "cache/",
    "metadata": "sentence",
    "split": "leave_one_out",
    "num_proc": 1,
    "max_item_seq_len": 50,

    # Sentence embeddings: statistics are fitted on training items only.
    "sent_emb_model": "Qwen/Qwen3-Embedding-4B",
    "sent_emb_model_paths": {
        "Qwen/Qwen3-Embedding-4B": "cache/models/Qwen/Qwen3-Embedding-4B",
        "sentence-transformers/sentence-t5-base": "cache/models/sentence-transformers/sentence-t5-base",
    },
    "sent_emb_dim": 2560,
    "sent_emb_pca": 128,
    "sent_emb_batch_size": 8,
    "sent_emb_max_length": 512,
    "sent_emb_dtype": "auto",  # BF16 on supported GPUs; FP32 on CPU.
    "sent_emb_device": None,  # Defaults to the training device.
    "sent_emb_local_files_only": True,
    "sent_emb_prompt": "",  # Item metadata is a document, not a search query.
    "sent_emb_normalize": True,

    # ---------------------- model parameters ----------------------
    # Both history encoders use causal attention.
    "encoder_type": "hstu",
    "supervision_mode": "all_positions",  # last_item supervises only the final valid label.
    "n_embd": 512,
    "n_layer": 4,
    "n_head": 4,
    "n_inner": 1024,
    "activation_function": "gelu_new",
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "layer_norm_epsilon": 1e-12,

    # HSTU: per-head U/V and Q/K dimensions, independent of n_embd/n_head.
    "hstu_hidden_dim": 64,
    "hstu_attention_dim": 64,
    "hstu_use_group_norm": True,
    "hstu_output_dropout": 0.1,
    "hstu_attention_dropout": 0.0,
    "hstu_attention_scaling": "input_length",  # input_length | max_seq_len

    # Diffusion / rectified flow and item retrieval
    "diffloss_w": 512,
    "diffloss_d": 3,
    "num_sampling_steps": 20,
    "diffusion_batch_mul": 1,
    "diff_temperature": 1.0,
    "grad_checkpointing": False,
    "tiger_uncond_prob": 0.1,
    "tiger_guidance_weight": 2.0,
    "use_rectified_flow": True,
    "rectified_flow_steps": 1000,
    "ode_solver": "euler",
    "temperature": 0.07,
    "tiger_knn_metric": "cosine",

    # ---------------------- optimization ----------------------
    "lr": 0.0005,
    "epochs": 100, # 150
    "weight_decay": 0.01,
    "train_batch_size": 256,
    "warmup_ratio": 0.1,  # Fraction of planned training epochs used for linear warmup.
    "warmup_steps": None,  # Optional fixed-step override; None uses warmup_ratio.
    "steps": None,
    "max_grad_norm": 1.0,
    "save_interval": 20,  # Overwrite latest every N epochs; always save on completion.

    # ---------------------- evaluation ----------------------
    "eval_batch_size": 64,
    "eval_interval": 1,
    "patience": 20,
    "topk": [5, 10],
    "metrics": ["ndcg", "recall"],
    "val_metric": "ndcg@10",

    # ---------------------- output and reproducibility ----------------------
    "run_id": "tigerdiff_sports_qwen3_hstu",
    "ckpt_dir": "ckpt/",
    "rand_seed": 2024,
    "reproducibility": True,
}
