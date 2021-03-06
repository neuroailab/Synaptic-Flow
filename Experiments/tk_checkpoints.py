import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *


def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    print("Loading {} dataset.".format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset)
    prune_loader = load.dataloader(
        dataset=args.dataset,
        batch_size=args.prune_batch_size,
        train=True,
        workers=args.workers,
        length=args.prune_dataset_ratio * num_classes,
        datadir=args.data_dir,
    )
    train_loader = load.dataloader(
        dataset=args.dataset,
        batch_size=args.train_batch_size,
        train=True,
        workers=args.workers,
        datadir=args.data_dir,
    )
    test_loader = load.dataloader(
        dataset=args.dataset,
        batch_size=args.test_batch_size,
        train=False,
        workers=args.workers,
        datadir=args.data_dir,
    )

    ## Model, Loss, Optimizer ##
    print("Creating {}-{} model.".format(args.model_class, args.model))
    if args.model in ["fc", "conv"]:
        norm_layer = load.norm_layer(args.norm_layer)
        print(f"Applying {args.norm_layer} normalization: {norm_layer}")
        model = load.model(args.model, args.model_class)(
            input_shape=input_shape,
            num_classes=num_classes,
            dense_classifier=args.dense_classifier,
            pretrained=args.pretrained,
            norm_layer=norm_layer,
        ).to(device)
    else:
        model = load.model(args.model, args.model_class)(
            input_shape=input_shape,
            num_classes=num_classes,
            dense_classifier=args.dense_classifier,
            pretrained=args.pretrained,
        ).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(
        generator.parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
        **opt_kwargs,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate
    )

    ## checkpointing setup ##
    if args.tk_steps_file is not None:
        save_steps = load.save_steps_file(args.tk_steps_file)
        steps_per_epoch = len(train_loader)
        max_epochs = int(save_steps[-1] / steps_per_epoch)
        print(f"Overriding train epochs to last step in file ")
        print(f"    pre_epochs set to 0, post_epochs set to {max_epochs}")
        setattr(args, "pre_epochs", 0)
        setattr(args, "post_epochs", max_epochs)
    else:
        save_steps = None

    ## Pre-Train ##
    print("Pre-Train for {} epochs.".format(args.pre_epochs))
    pre_result = train_eval_loop(
        model,
        loss,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        device,
        args.pre_epochs,
        args.verbose,
        save_steps=save_steps,
        save_freq=args.save_freq,
        save_path=args.save_path,
    )

    ## Prune ##
    if args.prune_epochs > 0:
        print("Pruning with {} for {} epochs.".format(args.pruner, args.prune_epochs))
        pruner = load.pruner(args.pruner)(
            generator.masked_parameters(
                model, args.prune_bias, args.prune_batchnorm, args.prune_residual
            )
        )
        sparsity = 10 ** (-float(args.compression))
        prune_loop(
            model,
            loss,
            pruner,
            prune_loader,
            device,
            sparsity,
            args.compression_schedule,
            args.mask_scope,
            args.prune_epochs,
            args.reinitialize,
        )

    ## Post-Train ##
    print("Post-Training for {} epochs.".format(args.post_epochs))
    post_result = train_eval_loop(
        model,
        loss,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        device,
        args.post_epochs,
        args.verbose,
        save_steps=save_steps,
        save_freq=args.save_freq,
        save_path=args.save_path,
    )

    ## Display Results ##
    frames = [
        pre_result.head(1),
        pre_result.tail(1),
        post_result.head(1),
        post_result.tail(1),
    ]
    train_result = pd.concat(frames, keys=["Init.", "Pre-Prune", "Post-Prune", "Final"])
    print("Train results:\n", train_result)
    if args.prune_epochs > 0:
        prune_result = metrics.summary(
            model,
            pruner.scores,
            metrics.flop(model, input_shape, device),
            lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual),
        )
        total_params = int((prune_result["sparsity"] * prune_result["size"]).sum())
        possible_params = prune_result["size"].sum()
        total_flops = int((prune_result["sparsity"] * prune_result["flops"]).sum())
        possible_flops = prune_result["flops"].sum()

        print("Prune results:\n", prune_result)
        print(
            "Parameter Sparsity: {}/{} ({:.4f})".format(
                total_params, possible_params, total_params / possible_params
            )
        )
        print(
            "FLOP Sparsity: {}/{} ({:.4f})".format(
                total_flops, possible_flops, total_flops / possible_flops
            )
        )

    ## Save Results and Model ##
    if args.save:
        print("Saving results.")
        pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        torch.save(model.state_dict(), "{}/model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(), "{}/optimizer.pt".format(args.result_dir))
        if args.prune_epochs > 0:
            prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
            torch.save(pruner.state_dict(), "{}/pruner.pt".format(args.result_dir))
