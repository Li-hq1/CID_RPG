import datetime
import logging
import yaml
from pathlib import Path

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M")
    return cur


def get_file_name(config: dict, suffix: str = ''):
    """Compatibility helper with a concise timestamp and task name."""
    timestamp = config.get("run_timestamp", config.get("run_local_time", "run"))
    return f"{timestamp}_{config['run_id']}{suffix}"


def experiment_dir(config: dict) -> Path:
    """Return the short, human-addressable directory for one experiment task."""
    return (
        Path(config["ckpt_dir"])
        / str(config["dataset"])
        / str(config["model"])
        / str(config["category"])
        / str(config["run_id"])
    )


def checkpoint_paths(config: dict) -> dict[str, Path]:
    root = experiment_dir(config)
    return {
        "root": root,
        "latest": root / f"latest_{config['run_id']}.pth",
        "best": root / f"best_{config['run_id']}.pth",
        # Compatibility alias: text logs now live directly under the experiment root.
        "logs": root,
        "log": root / f"{config.get('run_mode', 'train')}_{get_file_name(config, '.log')}",
    }


def init_logger(config: dict):
    paths = checkpoint_paths(config)
    paths["root"].mkdir(parents=True, exist_ok=True)
    logfilepath = paths["log"]

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    handlers = [sh]
    if config['accelerator'].is_main_process:
        fh = logging.FileHandler(logfilepath)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fileformatter)
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)

    if not config['accelerator'].is_main_process:
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()


def log(message, accelerator, logger, level='info'):
    if accelerator.is_main_process:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')


def _make_loggable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_make_loggable(v) for v in value]
    if isinstance(value, tuple):
        return [_make_loggable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_loggable(v) for k, v in value.items()}
    return str(value)


def config_for_log(config: dict) -> dict:
    config = config.copy()
    config.pop('device', None)
    config.pop('accelerator', None)
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = str(v)
    return config


def format_hyper_parameters(config: dict) -> str:
    loggable_config = {
        key: _make_loggable(value)
        for key, value in sorted(config.items())
        if key != 'accelerator'
    }
    return yaml.safe_dump(
        loggable_config,
        sort_keys=True,
        default_flow_style=False,
    ).rstrip()
