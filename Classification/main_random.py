import copy
import os
from collections import OrderedDict

import arg_parser
import dataset as dataset_utils
import evaluation
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate


def _get_dataset_targets(dataset):
    for attr in ("targets", "labels", "_labels"):
        if hasattr(dataset, attr):
            return np.array(getattr(dataset, attr))
    raise AttributeError("Dataset has no targets/labels/_labels attribute")


def _set_dataset_targets(dataset, targets):
    for attr in ("targets", "labels", "_labels"):
        if hasattr(dataset, attr):
            setattr(dataset, attr, targets.copy())
    return dataset


def _slice_dataset(dataset, mask):
    indices = np.flatnonzero(mask)
    sliced = copy.deepcopy(dataset)
    if hasattr(sliced, "data"):
        sliced.data = sliced.data[indices]
    if hasattr(sliced, "imgs"):
        sliced.imgs = sliced.imgs[indices]
    _set_dataset_targets(sliced, _get_dataset_targets(dataset)[indices])
    return sliced


def _build_incremental_unlearn_datasets(train_dataset, args):
    cumulative_classes = dataset_utils.parse_classes_to_replace(
        args.classes_to_replace
    )
    if args.class_to_replace < 0:
        raise ValueError(
            "--incremental_forget_only requires --class_to_replace to be a non-negative class id"
        )
    current_class = int(args.class_to_replace)
    if current_class not in cumulative_classes:
        raise ValueError(
            f"Current class {current_class} must be present in cumulative classes {cumulative_classes}"
        )

    targets = _get_dataset_targets(train_dataset)
    cumulative_mask = np.isin(targets, cumulative_classes)
    forget_mask = targets == current_class
    retain_mask = ~cumulative_mask
    excluded_mask = cumulative_mask & ~forget_mask

    forget_dataset = _slice_dataset(train_dataset, forget_mask)
    retain_dataset = _slice_dataset(train_dataset, retain_mask)

    old_forgotten = [cls for cls in cumulative_classes if cls != current_class]
    print(
        "incremental split: "
        f"current_forget_class={current_class} "
        f"cumulative_forgotten={cumulative_classes} "
        f"old_forgotten={old_forgotten}"
    )
    print(
        "incremental split counts: "
        f"forget={int(forget_mask.sum())} "
        f"retain={int(retain_mask.sum())} "
        f"excluded_old={int(excluded_mask.sum())}"
    )
    if len(forget_dataset) > 0:
        print(
            "forget labels:",
            np.unique(_get_dataset_targets(forget_dataset)).tolist(),
        )
    if len(retain_dataset) > 0:
        print(
            "retain labels:",
            np.unique(_get_dataset_targets(retain_dataset)).tolist(),
        )

    assert int(forget_mask.sum()) + int(retain_mask.sum()) + int(excluded_mask.sum()) == len(
        train_dataset
    )
    return forget_dataset, retain_dataset


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    if args.incremental_forget_only:
        forget_dataset, retain_dataset = _build_incremental_unlearn_datasets(
            train_loader_full.dataset, args
        )
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
    else:
        forget_dataset = copy.deepcopy(marked_loader.dataset)
        if args.dataset == "svhn":
            try:
                marked = forget_dataset.targets < 0
            except:
                marked = forget_dataset.labels < 0
            forget_dataset.data = forget_dataset.data[marked]
            try:
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
            except:
                forget_dataset.labels = -forget_dataset.labels[marked] - 1
            forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            try:
                marked = retain_dataset.targets >= 0
            except:
                marked = retain_dataset.labels >= 0
            retain_dataset.data = retain_dataset.data[marked]
            try:
                retain_dataset.targets = retain_dataset.targets[marked]
            except:
                retain_dataset.labels = retain_dataset.labels[marked]
            retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        else:
            try:
                marked = forget_dataset.targets < 0
                forget_dataset.data = forget_dataset.data[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(
                    forget_dataset, seed=seed, shuffle=True
                )
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.data = retain_dataset.data[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(
                    retain_dataset, seed=seed, shuffle=True
                )
                assert len(forget_dataset) + len(retain_dataset) == len(
                    train_loader_full.dataset
                )
            except:
                marked = forget_dataset.targets < 0
                forget_dataset.imgs = forget_dataset.imgs[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(
                    forget_dataset, seed=seed, shuffle=True
                )
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.imgs = retain_dataset.imgs[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(
                    retain_dataset, seed=seed, shuffle=True
                )
                assert len(forget_dataset) + len(retain_dataset) == len(
                    train_loader_full.dataset
                )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:

        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.mask_path:
            mask = torch.load(args.mask_path)

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args, mask)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
