"""Explicit TigerDiff experiment setup. The training loop is in engine/train.py."""
from datetime import datetime
from logging import getLogger

from accelerate import Accelerator
from torch.utils.data import DataLoader

from data.amazon_reviews import AmazonReviews2014
from data.tigerdiff_data import TigerDiffTokenizer
from models.tigerdiff import TigerDiff
from engine.train import train
from engine.evaluate import evaluate
from engine.checkpoint import load_checkpoint, prepare_test_model, resolve_checkpoint
from misc.config import parse_config
from misc.logging import init_logger, get_local_time, checkpoint_paths, format_hyper_parameters, log
from misc.seed import init_device, init_seed


def main():
    config, args = parse_config()
    # ----------------------------- initialize -------------------------------
    # init device and run mode
    config["device"], config["use_ddp"] = init_device()
    config["run_local_time"] = get_local_time()
    config["run_timestamp"] = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # init run mode
    config["run_mode"] = "eval" if args.eval_only else "resume" if args.resume else "train"
    # init checkpoint paths
    paths = checkpoint_paths(config)
    selected_checkpoint = args.checkpoint
    if selected_checkpoint is None and (args.resume or args.eval_only):
        selected_checkpoint = paths["latest"]
    if selected_checkpoint is not None:
        selected_checkpoint = resolve_checkpoint(selected_checkpoint)
    project_dir = str(paths["root"] / "tensorboard")
    accelerator = Accelerator(log_with="tensorboard", project_dir=project_dir)
    config["accelerator"] = accelerator
    # init seed and logger
    init_seed(config["rand_seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    log(f"Hyper Parameters:\n{format_hyper_parameters(config)}", accelerator, logger)
    log(f"Device: {config['device']}", accelerator, logger)
    log(f"Experiment directory: {paths['root']}", accelerator, logger)

    # ----------------------------- load dataset -------------------------------
    dataset = AmazonReviews2014(config)
    log(dataset, accelerator, logger)
    splits = dataset.split()
    # Only the first process creates embedding/sample caches; others reuse them.
    with accelerator.main_process_first():
        tokenizer = TigerDiffTokenizer(config, dataset)
        tokenized = tokenizer.tokenize(splits)
    if args.smoke_test:
        tokenized = {split: samples.select(range(min(64, len(samples)))) for split, samples in tokenized.items()}

    # ----------------------------- load model -------------------------------
    with accelerator.main_process_first():
        model = TigerDiff(config, dataset, tokenizer)
        if selected_checkpoint is not None:
            load_checkpoint(model, selected_checkpoint, config["device"])
            log(f"Loaded model checkpoint from {selected_checkpoint}", accelerator, logger)
    log(model, accelerator, logger)
    log(model.n_parameters, accelerator, logger)


    # ----------------------------- train or evaluate -------------------------------
    test_loader = DataLoader(tokenized["test"], batch_size=config["eval_batch_size"], shuffle=False)
    
    try:
        best_epoch, best_val_score = None, None
        if args.eval_only:
            test_checkpoint = selected_checkpoint
        else:
            train_loader = DataLoader(tokenized["train"], batch_size=config["train_batch_size"], shuffle=True)
            val_loader = DataLoader(tokenized["val"], batch_size=config["eval_batch_size"], shuffle=False)
            model, best_epoch, best_val_score = train(
                model, train_loader, val_loader, config, tokenizer, accelerator, paths["latest"],
                resume_checkpoint=selected_checkpoint if args.resume else None,
            )
            accelerator.wait_for_everyone()
            # Zero-step / already-completed resume runs may not write a new checkpoint.
            test_checkpoint = paths["latest"] if best_epoch is not None and paths["latest"].is_file() else selected_checkpoint
            if test_checkpoint is None:
                raise RuntimeError("No checkpoint was saved. Enable training or use --eval-only with a checkpoint.")
        model, test_loader = prepare_test_model(model, test_loader, accelerator, test_checkpoint)
        log(f"Testing checkpoint: {test_checkpoint}", accelerator, logger)
        results = evaluate(model, test_loader, config, tokenizer, accelerator)
        if accelerator.is_main_process:
            accelerator.log({f"Test_Metric/{key}": value for key, value in results.items()})
        excel_row = "\t".join(
            f"{results[key]:.3f}" for key in ("recall@5", "ndcg@5", "recall@10", "ndcg@10")
        )
        log(f"Test Results: {results}\nR@5\tN@5\tR@10\tN@10\n{excel_row}", accelerator, logger)
        return {"best_epoch": best_epoch, "best_val_score": best_val_score, "test_results": results}
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
