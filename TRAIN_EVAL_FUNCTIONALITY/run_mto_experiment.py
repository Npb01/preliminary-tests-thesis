"""
Entry-point for reproducing the SuTraN+ multi-task experiments.

This wrapper exposes a tiny, user-facing API that only requires the
event log identifier, the desired MTO technique, and the seed. All
other log-specific and technique-specific hyperparameters are pulled
from `log_configs.py` and `technique_configs.py`, matching the settings
reported in the paper.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict

from TRAIN_EVAL_FUNCTIONALITY import log_configs, technique_configs
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_EQUAL_WEIGHTING import (
    train_eval as train_eval_equal_weighting,
)
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_GRADNORM import (
    train_eval as train_eval_gradnorm,
)
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_PCGRAD import (
    train_eval as train_eval_pcgrad,
)
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_UW import (
    train_eval as train_eval_uw,
)

# ---------------------------------------------------------------------------
# Technique dispatch and configuration
# ---------------------------------------------------------------------------

_TECHNIQUE_DISPATCH: Dict[str, Callable[..., None]] = {
    "equal_weighting": train_eval_equal_weighting,
    "gradnorm": train_eval_gradnorm,
    "uw": train_eval_uw,
    "uw_plus": train_eval_uw,
    "pcgrad": train_eval_pcgrad,
}

_TECHNIQUE_CONFIGS: Dict[str, Dict[str, object]] = {
    "equal_weighting": {},
    "gradnorm": technique_configs.gradnorm_config_dict,
    "uw": technique_configs.uw_config_dict,
    "uw_plus": technique_configs.uw_plus_config_dict,
    "pcgrad": {"lr_model": 0.0002},
}


def run_mto_experiment(log_name: str,
                       mto_technique: str,
                       seed: int,
                       subset_fraction: float= 1.0,
                       lambda_ltn: float = 0.0,
                       detach_mode: str = "none",
                       balance_losses: bool = False,
                       scale_ttne: float = 1.0,
                       scale_rrt: float = 1.0,
                       val_subset_fraction: float = 1.0,
                       num_epochs: int = 200,
                       patience: int = 24) -> None:
    """
    Train and evaluate SuTraN+ with the specified MTO technique.

    Parameters
    ----------
    log_name : {'BPIC_17', 'BPIC_17_DR', 'BPIC_19'}
        Identifier of the event log to use.
    mto_technique : {'equal_weighting', 'gradnorm', 'uw', 'uw_plus', 'pcgrad'}
        Multi-task optimisation strategy to deploy.
    seed : int
        Random seed (1..5 in the paper experiments).
    subset_fraction : float, default=1.0 (optional)
        Fraction of the dataset to use for training.
    lambda_ltn : float, default=0.0 (optional)
        Weight of the LTN cross-task consistency loss term.
    detach_mode : str, default="none" (optional)
        Which side of the LTN consistency axiom to detach (freeze).
    balance_losses : bool, default=False (optional)
        Apply static rescaling to equalize ttne/rrt loss magnitudes.
    scale_ttne : float, default=1.0 (optional)
        Static rescaling factor for the ttne loss (only used if --balance_losses).
    scale_rrt : float, default=1.0 (optional)
        Static rescaling factor for the rrt loss (only used if --balance_losses).
    val_subset_fraction : float, default=1.0 (optional)
        Fraction of the validation set (case-level) to use for per-epoch
        validation. Does not affect the final test-set evaluation.
    num_epochs : int, default=200 (optional)
        Number of epochs to train for.
    patience : int, default=24 (optional)
        Patience for early stopping.
    """

    log_key = log_name.upper()
    if log_key not in log_configs.log_name_list:
        raise ValueError(
            f"'log_name' must be one of {log_configs.log_name_list}, got '{log_name}'."
        )

    technique_key = mto_technique.lower()
    if technique_key not in _TECHNIQUE_DISPATCH:
        raise ValueError(
            f"Unsupported MTO technique '{mto_technique}'. "
            "Choose from 'equal_weighting', 'gradnorm', 'uw', 'uw_plus', or 'pcgrad'."
        )

    train_fn = _TECHNIQUE_DISPATCH[technique_key]
    technique_kwargs = dict(_TECHNIQUE_CONFIGS[technique_key])  # copy

    # Gather log-specific parameters.
    base_kwargs = {
        "log_name": log_key,
        "median_caselen": log_configs.median_caselen_dict[log_key],
        "outcome_bool": log_configs.outcome_bools_dict[log_key],
        "out_mask": log_configs.out_masks_dict[log_key],
        "clen_dis_ref": True,
        "out_type": log_configs.out_types_dict[log_key],
        "num_outclasses": log_configs.num_outclasses_dict[log_key],
        "seed": seed,
        "out_string": None,
        "subset_fraction": subset_fraction,
        "lambda_ltn": lambda_ltn,
        "detach_mode": detach_mode,
        "balance_losses": balance_losses,
        "scale_ttne": scale_ttne,
        "scale_rrt": scale_rrt,
        "val_subset_fraction": val_subset_fraction,
        "num_epochs": num_epochs,
        "patience": patience,  
    }

    base_kwargs.update(technique_kwargs)

    train_fn(**base_kwargs)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate SuTraN+ with a selected multi-task optimisation technique."
        )
    )
    parser.add_argument(
        "--log_name",
        required=True,
        choices=log_configs.log_name_list,
        help="Event log identifier.",
    )
    parser.add_argument(
        "--MTO_technique",
        required=True,
        choices=list(_TECHNIQUE_DISPATCH.keys()),
        help="Multi-task optimisation strategy to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed (1..5 in the paper experiments).",
    )
    parser.add_argument(
        "--subset_fraction",
        type=float,
        required=False,
        default=1.0,
        help="Fraction of the dataset to use for training.",
    )
    parser.add_argument(
        "--lambda_ltn",
        type=float,
        required=False,
        default=0.0,
        help="Weight of the LTN cross-task consistency loss term.",
    )
    parser.add_argument(
        "--detach_mode",
        type=str,
        required=False,
        default="none",
        choices=["none", "ttne", "rrt"],
        help="Which side of the LTN consistency axiom to detach (freeze)."
    )
    parser.add_argument(
        "--balance_losses",
        action="store_true",
        help="Apply static rescaling to equalize ttne/rrt loss magnitudes."
    )
    parser.add_argument(
        "--scale_ttne",
        type=float,
        required=False,
        default=1.0,
        help="Static rescaling factor for the ttne loss (only used if --balance_losses)."
    )
    parser.add_argument(
        "--scale_rrt",
        type=float,
        required=False,
        default=1.0,
        help="Static rescaling factor for the rrt loss (only used if --balance_losses)."
    )
    parser.add_argument(
        "--val_subset_fraction",
        type=float,
        required=False,
        default=1.0,
        help="Fraction of the validation set (case-level) to use for per-epoch validation."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=200,
        help="Number of epochs to train for."
    )
    parser.add_argument(
        "--patience",
        type=int,
        required=False,
        default=24,
        help="Patience for early stopping."
    )
    return parser


if __name__ == "__main__":
    arguments = _build_arg_parser().parse_args()
    run_mto_experiment(
        log_name=arguments.log_name,
        mto_technique=arguments.MTO_technique,
        seed=arguments.seed,
        subset_fraction=arguments.subset_fraction,
        lambda_ltn=arguments.lambda_ltn,
        detach_mode=arguments.detach_mode,
        balance_losses=arguments.balance_losses,
        scale_ttne=arguments.scale_ttne,
        scale_rrt=arguments.scale_rrt,
        val_subset_fraction=arguments.val_subset_fraction,
        num_epochs=arguments.num_epochs,
        patience=arguments.patience,
    )
