import argparse
import copy
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

try:
    from private_model_template import build_model
except ImportError as exc:
    raise ImportError(
        "Missing private_model_template.py. This repository intentionally omits the\n"
        "proprietary model implementation. Create private_model_template.py with a\n"
        "build_model() function that returns a torch.nn.Module."
    ) from exc


class CustomDataset(Dataset):
    def __init__(self, features: pd.DataFrame, labels: np.ndarray):
        self.features = torch.tensor(features.values.astype(np.float32))
        self.labels = torch.tensor(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_metrics(labels, predictions, scores):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    scores = np.asarray(scores)

    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    return {
        "acc": accuracy_score(labels, predictions),
        "spe": specificity,
        "sen": recall_score(labels, predictions, zero_division=0),
        "pre": precision_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "auc": roc_auc_score(labels, scores),
        "pr_auc": auc(recall_curve, precision_curve),
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "fpr": fpr,
        "tpr": tpr,
    }


def evaluate_model(model, data_loader, criterion, device, threshold=0.5):
    model.eval()
    eval_loss = 0.0
    all_scores = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            scores = model(features.unsqueeze(1)).view(-1)
            eval_loss += criterion(scores, labels).item()

            predictions = (scores > threshold).int()
            all_scores.extend(scores.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    metrics = compute_metrics(all_labels, all_predictions, all_scores)
    metrics["loss"] = eval_loss / max(len(data_loader), 1)
    return metrics


def save_curve_csv(output_path: Path, x_name: str, y_name: str, x_values, y_values) -> None:
    pd.DataFrame({x_name: x_values, y_name: y_values}).to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Open-source training/evaluation pipeline. The proprietary model implementation "
            "is intentionally excluded."
        )
    )
    parser.add_argument("--train-neg", type=str, default="../embeddings_ProtT5/Train_nonPSP_for_PdPS.csv")
    parser.add_argument("--train-pos", type=str, default="../embeddings_ProtT5/PdPS.csv")
    parser.add_argument("--test-neg", type=str, default="../embeddings_ProtT5/NoPS-test.csv")
    parser.add_argument("--test-pos", type=str, default="../embeddings_ProtT5/PdPS-test.csv")
    parser.add_argument("--output-dir", type=str, default="../save")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-positive-samples", type=int, default=194)
    parser.add_argument("--train-negative-samples", type=int, default=388)
    parser.add_argument("--val-positive-samples", type=int, default=20)
    parser.add_argument("--val-negative-samples", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    set_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "model"
    roc_dir = output_dir / "ROC"
    pr_dir = output_dir / "PR"
    ensure_dir(model_dir)
    ensure_dir(roc_dir)
    ensure_dir(pr_dir)

    train_data_neg = pd.read_csv(args.train_neg)
    train_data_pos = pd.read_csv(args.train_pos)

    val_data_pos = train_data_pos.iloc[: args.val_positive_samples]
    val_data_neg = train_data_neg.iloc[: args.val_negative_samples]
    train_data_pos = train_data_pos.iloc[args.val_positive_samples :]
    train_data_neg = train_data_neg.iloc[args.val_negative_samples :]

    fold_indices = [
        list(range(i * args.train_negative_samples, (i + 1) * args.train_negative_samples))
        for i in range(args.num_folds)
    ]

    all_fold_metric_lists = {key: [] for key in ["acc", "spe", "sen", "pre", "f1", "mcc", "auc", "pr_auc"]}
    roc_fprs, roc_tprs, pr_precisions, pr_recalls = [], [], [], []

    for fold, val_index in enumerate(fold_indices):
        print(f"===== Fold {fold + 1}/{args.num_folds} =====")

        train_positive_samples = train_data_pos
        train_positive_labels = np.ones(args.train_positive_samples)
        train_negative_samples = train_data_neg.iloc[val_index]
        train_negative_labels = np.zeros(args.train_negative_samples)

        train_data = pd.concat([train_positive_samples, train_negative_samples], axis=0)
        train_labels = np.concatenate([train_positive_labels, train_negative_labels])

        val_data = pd.concat([val_data_pos, val_data_neg], axis=0)
        val_labels = np.concatenate(
            [np.ones(args.val_positive_samples), np.zeros(args.val_negative_samples)]
        )

        train_dataset = CustomDataset(train_data, train_labels)
        val_dataset = CustomDataset(val_data, val_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = build_model().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

        best_mcc = float("-inf")
        best_state_dict = None
        fold_history = {key: [] for key in ["acc", "spe", "sen", "pre", "f1", "mcc", "auc", "pr_auc"]}

        for epoch in range(args.num_epochs):
            model.train()
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                scores = model(features.unsqueeze(1)).view(-1)
                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()

            val_metrics = evaluate_model(model, val_loader, criterion, device, args.threshold)
            for key in fold_history:
                fold_history[key].append(val_metrics[key])

            # Save mean curves across epochs/folds in the same spirit as the original script.
            sampled_x = np.linspace(0, 1, 100)
            sampled_recall = np.interp(sampled_x, val_metrics["precision_curve"], val_metrics["recall_curve"])
            sampled_tpr = np.interp(sampled_x, val_metrics["fpr"], val_metrics["tpr"])
            pr_precisions.append(sampled_x)
            pr_recalls.append(sampled_recall)
            roc_fprs.append(sampled_x)
            roc_tprs.append(sampled_tpr)

            if val_metrics["mcc"] > best_mcc:
                best_mcc = val_metrics["mcc"]
                best_state_dict = copy.deepcopy(model.state_dict())

        if best_state_dict is None:
            raise RuntimeError("Training finished without a best checkpoint.")

        checkpoint_path = model_dir / f"public_train_fold_{fold}.pt"
        torch.save(best_state_dict, checkpoint_path)

        fold_means = {key: float(np.mean(values)) for key, values in fold_history.items()}
        print(
            "{acc:.4f}  {spe:.4f}  {sen:.4f}  {pre:.4f}  {f1:.4f}  {mcc:.4f}  {auc:.4f}  {pr_auc:.4f}".format(
                **fold_means
            )
        )

        for key, value in fold_means.items():
            all_fold_metric_lists[key].append(value)

    print(
        "Overall: {acc:.4f}  {spe:.4f}  {sen:.4f}  {pre:.4f}  {f1:.4f}  {mcc:.4f}  {auc:.4f}  {pr_auc:.4f}".format(
            **{key: float(np.mean(values)) for key, values in all_fold_metric_lists.items()}
        )
    )

    mean_precision = np.mean(pr_precisions, axis=0)
    mean_recall = np.mean(pr_recalls, axis=0)
    mean_fpr = np.mean(roc_fprs, axis=0)
    mean_tpr = np.mean(roc_tprs, axis=0)

    save_curve_csv(pr_dir / "train_pr_mean.csv", "Precision", "Recall", mean_precision, mean_recall)
    save_curve_csv(roc_dir / "train_roc_mean.csv", "FPR", "TPR", mean_fpr, mean_tpr)

    print("##### Test #####")
    test_neg = pd.read_csv(args.test_neg)
    test_pos = pd.read_csv(args.test_pos)
    test_data = pd.concat([test_neg, test_pos], axis=0)
    test_labels = np.concatenate([np.zeros(test_neg.shape[0]), np.ones(test_pos.shape[0])])

    test_dataset = CustomDataset(test_data, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    all_fold_scores = []
    for fold in range(args.num_folds):
        print(f"--- Fold {fold + 1} ---")
        checkpoint_path = model_dir / f"public_train_fold_{fold}.pt"
        model = build_model().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        fold_scores = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                scores = model(features.unsqueeze(1)).view(-1)
                fold_scores.extend(scores.detach().cpu().tolist())

        fold_scores = np.asarray(fold_scores)
        all_fold_scores.append(fold_scores)
        fold_predictions = (fold_scores > args.threshold).astype(int)
        fold_metrics = compute_metrics(test_labels, fold_predictions, fold_scores)

        save_curve_csv(roc_dir / f"test_roc_fold_{fold + 1}.csv", "FPR", "TPR", fold_metrics["fpr"], fold_metrics["tpr"])
        save_curve_csv(
            pr_dir / f"test_pr_fold_{fold + 1}.csv",
            "Precision",
            "Recall",
            fold_metrics["precision_curve"],
            fold_metrics["recall_curve"],
        )

        print("ACC Spe Sen Pre F1 MCC AUC PR_AUC")
        print(
            f"{fold_metrics['acc']:.4f} {fold_metrics['spe']:.4f} {fold_metrics['sen']:.4f} "
            f"{fold_metrics['pre']:.4f} {fold_metrics['f1']:.4f} {fold_metrics['mcc']:.4f} "
            f"{fold_metrics['auc']:.4f} {fold_metrics['pr_auc']:.4f}"
        )

    mean_scores = np.mean(np.asarray(all_fold_scores), axis=0)
    final_predictions = (mean_scores > args.threshold).astype(int)
    final_metrics = compute_metrics(test_labels, final_predictions, mean_scores)

    save_curve_csv(roc_dir / "test_roc_avg.csv", "FPR", "TPR", final_metrics["fpr"], final_metrics["tpr"])
    save_curve_csv(
        pr_dir / "test_pr_avg.csv",
        "Precision",
        "Recall",
        final_metrics["precision_curve"],
        final_metrics["recall_curve"],
    )

    print("##### Final Average Results #####")
    print("ACC Spe Sen Pre F1 MCC AUC PR_AUC")
    print(
        f"{final_metrics['acc']:.4f} {final_metrics['spe']:.4f} {final_metrics['sen']:.4f} "
        f"{final_metrics['pre']:.4f} {final_metrics['f1']:.4f} {final_metrics['mcc']:.4f} "
        f"{final_metrics['auc']:.4f} {final_metrics['pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
