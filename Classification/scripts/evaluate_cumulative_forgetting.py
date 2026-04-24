#!/usr/bin/env python
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models import model_dict
from utils import NormalizeByChannelMeanStd, setup_seed


def parse_classes(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cumulative class forgetting")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--forgotten_classes", required=True, help="Comma-separated class ids, e.g. 0,1,2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data", default="../data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)
    return parser.parse_args()


def load_model(args, device):
    if args.dataset != "cifar10":
        raise ValueError("This evaluator currently supports CIFAR-10 only.")

    model = model_dict[args.arch](num_classes=args.num_classes)
    model.normalize = NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


def build_test_loader(args):
    test_set = CIFAR10(
        args.data,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=True,
    )
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def evaluate(model, loader, args, device):
    correct = np.zeros(args.num_classes, dtype=np.int64)
    total = np.zeros(args.num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images).argmax(dim=1)
            for class_id in range(args.num_classes):
                mask = targets == class_id
                total[class_id] += int(mask.sum().item())
                correct[class_id] += int((preds[mask] == targets[mask]).sum().item())

    per_class = {}
    for class_id in range(args.num_classes):
        per_class[class_id] = None
        if total[class_id] > 0:
            per_class[class_id] = 100.0 * correct[class_id] / total[class_id]

    forgotten = parse_classes(args.forgotten_classes)
    retain = [class_id for class_id in range(args.num_classes) if class_id not in forgotten]

    def grouped_accuracy(classes):
        grouped_total = int(total[classes].sum()) if classes else 0
        if grouped_total == 0:
            return None
        grouped_correct = int(correct[classes].sum())
        return 100.0 * grouped_correct / grouped_total

    full_total = int(total.sum())
    full_accuracy = 100.0 * int(correct.sum()) / full_total if full_total else None
    forget_accuracy = grouped_accuracy(forgotten)
    retain_accuracy = grouped_accuracy(retain)

    return {
        "model_path": args.model_path,
        "forgotten_classes": ",".join(str(item) for item in forgotten),
        "forget_accuracy": forget_accuracy,
        "UA": None if forget_accuracy is None else 100.0 - forget_accuracy,
        "retain_accuracy": retain_accuracy,
        "full_test_accuracy": full_accuracy,
        "per_class_accuracy": per_class,
    }


def write_csv(result, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model_path": result["model_path"],
        "forgotten_classes": result["forgotten_classes"],
        "forget_accuracy": result["forget_accuracy"],
        "UA": result["UA"],
        "retain_accuracy": result["retain_accuracy"],
        "full_test_accuracy": result["full_test_accuracy"],
    }
    for class_id, accuracy in result["per_class_accuracy"].items():
        row[f"class_{class_id}_accuracy"] = accuracy

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    setup_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    model = load_model(args, device)
    loader = build_test_loader(args)
    result = evaluate(model, loader, args, device)
    write_csv(result, args.output)

    print("Cumulative forgetting evaluation")
    for key in ["model_path", "forgotten_classes", "forget_accuracy", "UA", "retain_accuracy", "full_test_accuracy"]:
        print(f"{key}: {result[key]}")
    for class_id, accuracy in result["per_class_accuracy"].items():
        print(f"class_{class_id}_accuracy: {accuracy}")


if __name__ == "__main__":
    main()
