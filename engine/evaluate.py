from collections import defaultdict, OrderedDict

import torch
from tqdm import tqdm

from engine.metrics import Evaluator


def evaluate(model, dataloader, config, tokenizer, accelerator, split="test"):
    model.eval()
    evaluator = Evaluator(config, tokenizer)
    results_by_batch = defaultdict(list)
    generation_model = accelerator.unwrap_model(model)
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Eval - {split}"):
        with torch.no_grad():
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}
            preds = generation_model.generate(batch, n_return_sequences=evaluator.maxk)
            labels = batch["labels"]
            if config["use_ddp"]:
                preds, labels = accelerator.gather_for_metrics((preds, labels))
            results = evaluator.calculate_metrics(preds, labels)
            for key, value in results.items():
                results_by_batch[key].append(value)
    if not results_by_batch:
        raise ValueError(f"The {split} dataloader is empty.")
    return OrderedDict(
        (key, torch.cat(values).mean().item())
        for key, values in results_by_batch.items()
    )
