# From V-GPS (https://github.com/nakamotoo/V-GPS), jaxrl_m of bridge_data_v2 (https://github.com/rail-berkeley/bridge_data_v2), MIT. Modified for this thesis.
import datetime
import tempfile
from copy import copy
from socket import gethostname

import absl.flags as flags
import ml_collections
import wandb


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


class WandBLogger(object):
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "jaxrl_m"  # WandB Project Name
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        # Which entity to log as (default: your own user)
        config.exp_descriptor = ""  # Run name (doesn't have to be unique)
        # Unique identifier for run (will be automatically generated unless
        # provided)
        config.unique_identifier = ""
        config.group = None
        return config

    # TODO(reproducibility): end-to-end experiment reproducibility from a
    # wandb run id. Goal: `python train_*.py --resume_from_run <RUN_ID>` should
    # reproduce the exact past experiment — config, seed, data mix, all flags —
    # by pulling them from that run's wandb.config, with no dependence on the
    # current state of `experiments/configs/*.py` or the sbatch script. Run id
    # in → identical experiment config out. Also: associate checkpoints with
    # the run id on disk so weights and config stay linked.
    #
    # Current state: create_resume() below only continues logging into the same
    # run; FLAGS.config is still parsed locally, so editing a config file
    # silently changes any "resumed" run. Not reproducible.

    @staticmethod
    def create_resume(run_id, variant=None, debug=False):
        """
        Create WandBLogger that resumes an existing wandb run.

        Args:
            run_id: Full wandb run ID (e.g., "username/project/run_id")
            variant: Optional dict with additional config to log
            debug: If True, run in disabled mode

        Returns:
            WandBLogger instance wrapping the resumed run
        """
        # Load existing run from wandb API
        api = wandb.Api()
        wandb_run = api.run(run_id)

        # Create minimal config for resume
        config = ml_collections.ConfigDict()
        config.project = wandb_run.project
        config.entity = wandb_run.entity
        config.experiment_id = wandb_run.id
        config.experiment_descriptor = wandb_run.name
        config.unique_identifier = wandb_run.id
        config.group = wandb_run.group

        # Create logger instance (bypasses normal __init__ with resume-specific logic)
        logger = object.__new__(WandBLogger)
        logger.config = config
        logger._variant = copy(variant) if variant is not None else {}

        mode = "disabled" if debug else "online"

        # Resume the wandb run
        logger.run = wandb.init(
            project=wandb_run.project,
            id=wandb_run.id,
            entity=wandb_run.entity,
            resume="must",
            mode=mode,
        )

        print(f"✓ Resumed wandb run: {run_id}")
        print(f"  Run name: {wandb_run.name}")
        print(f"  Save dir: {wandb_run.config.get('save_dir', 'N/A')}")

        # Store reference to original wandb run for config access
        logger._wandb_run = wandb_run

        return logger

    def get_resumed_config(self, key=None, default=None):
        """
        Get config value from resumed wandb run.

        Args:
            key: Config key to retrieve. If None, returns full config dict.
            default: Default value if key not found.

        Returns:
            Config value or full config dict
        """
        if hasattr(self, '_wandb_run'):
            if key is None:
                return self._wandb_run.config
            return self._wandb_run.config.get(key, default)
        return default

    def __init__(
        self,
        wandb_config,
        variant,
        wandb_output_dir=None,
        debug=False,
    ):
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        self.config.experiment_id = (
            self.experiment_id
        ) = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"  # NOQA

        print(self.config)

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if debug:
            mode = "disabled"
        else:
            mode = "online"

        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            save_code=True,
            mode=mode,
        )

        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        else:
            flag_dict = {}
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()
        wandb.config.update(flag_dict)

    def log(self, data: dict, step: int = None):
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        wandb.log(data, step=step)
