import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.general_recommender.bpr import BPR
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from bpr_regularized import RegularizedBPR


ROOT = Path(__file__).resolve().parents[1]


def check_regularization(config, dataset):
    original = BPR(config, dataset).to(config["device"])
    modified = RegularizedBPR(config, dataset).to(config["device"])

    # Compare models with exactly the same parameter values.
    modified.load_state_dict(original.state_dict())

    # Synthetic batch with repeated IDs to check occurrence weighting.
    batch = Interaction({
        original.USER_ID: torch.tensor([1, 1, 2]),
        original.ITEM_ID: torch.tensor([1, 2, 3]),
        original.NEG_ITEM_ID: torch.tensor([3, 3, 1]),
    }).to(config["device"])

    users = batch[original.USER_ID]
    positives = batch[original.ITEM_ID]
    negatives = batch[original.NEG_ITEM_ID]
    batch_size = users.numel()

    original.zero_grad(set_to_none=True)
    base_loss = original.calculate_loss(batch)
    base_loss.backward()
    original_parameters = dict(original.named_parameters())

    # Independent reference calculation of squared L2 penalty.
    with torch.no_grad():
        expected_reg = (
            original.user_embedding(users).square().sum()
            + original.item_embedding(positives).square().sum()
            + original.item_embedding(negatives).square().sum()
        ) / (2 * batch_size)

    # Analytical regularization gradient:
    # occurrence_count * embedding / batch_size.
    counts = {
        "user_embedding.weight": torch.bincount(
            users, minlength=original.n_users
        ),
        "item_embedding.weight": torch.bincount(
            torch.cat([positives, negatives]),
            minlength=original.n_items,
        ),
    }

    checks = []
    for coefficient in [0.0, 0.1, 0.5]:
        modified.reg_lambda = coefficient
        modified.zero_grad(set_to_none=True)

        total_loss = modified.calculate_loss(batch)
        expected_loss = base_loss.detach() + coefficient * expected_reg

        torch.testing.assert_close(
            total_loss, expected_loss, rtol=1e-5, atol=1e-7
        )
        total_loss.backward()

        for name, parameter in modified.named_parameters():
            reference = original_parameters[name]
            expected_gradient = (
                reference.grad
                + coefficient
                * counts[name].to(reference.dtype).unsqueeze(1)
                * reference.detach()
                / batch_size
            )
            torch.testing.assert_close(
                parameter.grad,
                expected_gradient,
                rtol=1e-5,
                atol=1e-7,
            )

        record = {
            "reg_lambda": coefficient,
            "ranking_loss": base_loss.item(),
            "regularization_before_lambda": expected_reg.item(),
            "total_loss": total_loss.item(),
            "loss_and_gradient_check": "passed",
        }
        checks.append(record)
        print(json.dumps(record, indent=2), flush=True)

    print("BPR REGULARIZATION CHECKS PASSED", flush=True)
    return checks


def main():
    torch.set_num_threads(4)

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir = ROOT / "results" / "bpr_smoke" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    # Preserve the exact source files used in this run.
    for relative_path in [
        "configs/common.yaml",
        "configs/bpr.yaml",
        "scripts/bpr_regularized.py",
        "scripts/check_bpr.py",
    ]:
        source = ROOT / relative_path
        shutil.copy2(source, output_dir / source.name)

    config = Config(
        model="BPR",
        dataset="ml-100k",
        config_file_list=[
            str(ROOT / "configs/common.yaml"),
            str(ROOT / "configs/bpr.yaml"),
        ],
        config_dict={
            "epochs": 2,
            "model_implementation": "RegularizedBPR",
            "checkpoint_dir": str(
                ROOT / "saved" / "bpr_smoke" / run_id
            ),
        },
    )

    settings = {
        "mode": "smoke",
        "model_implementation": "RegularizedBPR",
        "dataset": config["dataset"],
        "embedding_size": config["embedding_size"],
        "reg_lambda": config["reg_lambda"],
        "weight_decay": config["weight_decay"],
        "seed": config["seed"],
        "epochs": config["epochs"],
        "checkpoint_dir": config["checkpoint_dir"],
    }
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = logging.getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    checks = check_regularization(config, train_data._dataset)
    (output_dir / "regularization_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )

    # Reinitialize after the checks so they do not affect model initialization.
    init_seed(config["seed"], config["reproducibility"])
    model = RegularizedBPR(config, train_data._dataset).to(config["device"])
    logger.info(model)

    trainer = Trainer(config, model)
    try:
        _, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=False,
        )
        test_result = trainer.evaluate(
            test_data,
            load_best_model=True,
            show_progress=False,
        )
    finally:
        trainer.tensorboard.close()

    row = {
        "run_id": run_id,
        "mode": "smoke",
        "model": "RegularizedBPR",
        "embedding_size": config["embedding_size"],
        "reg_lambda": config["reg_lambda"],
        "seed": config["seed"],
        "epochs": config["epochs"],
    }
    for split, metrics in [
        ("valid", best_valid_result),
        ("test", test_result),
    ]:
        for metric, value in metrics.items():
            row[f"{split}_{metric}"] = float(value)

    pd.DataFrame([row]).to_csv(output_dir / "metrics.csv", index=False)
    logger.info("Best validation: %s", best_valid_result)
    logger.info("Test result: %s", test_result)

    print(json.dumps(row, indent=2), flush=True)
    print(f"Results: {output_dir}", flush=True)
    print(f"Checkpoint: {trainer.saved_model_file}", flush=True)
    print("BPR SMOKE TEST COMPLETED", flush=True)


if __name__ == "__main__":
    main()
