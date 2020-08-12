import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *

# TPU
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def run(args):
    if args.tpu:
        print_fn = xm.master_print
    else:
        print_fn = print
    ## Random Seed and Device ##
    torch.manual_seed(args.seed) 
    device = load.device(args.gpu, tpu=args.tpu)

    ## Data ##
    print_fn("Loading {} dataset.".format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset)
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
    print_fn("Creating {}-{} model.".format(args.model_class, args.model))
    if args.model in ["fc", "conv"]:
        norm_layer = load.norm_layer(args.norm_layer)
        print_fn(f"Applying {args.norm_layer} normalization: {norm_layer}")
        model = load.model(args.model, args.model_class)(
            input_shape=input_shape,
            num_classes=num_classes,
            dense_classifier=args.dense_classifier,
            pretrained=args.pretrained,
            norm_layer=norm_layer,
        )
    else:
        model = load.model(args.model, args.model_class)(
            input_shape=input_shape,
            num_classes=num_classes,
            dense_classifier=args.dense_classifier,
            pretrained=args.pretrained,
        )

    if args.tpu:
        # Model wrapper:
        # For the MP approach: nothing
        # For the DP approach: (this would also change how data is fed in train loop)
        # model_parallel = dp.DataParallel(model, device_ids=devices)

        # LR Rescale
        args.lr *= xm.xrt_world_size()
    model = model.to(device)

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
    assert args.tk_steps_file is not None
    save_steps = load.save_steps_file(args.tk_steps_file)
    steps_per_epoch = len(train_loader)
    max_epochs = int(save_steps[-1] / steps_per_epoch)
    print_fn(f"Overriding post_epochs to last step in file ")
    print_fn(f"    post_epochs set to {max_epochs}")
    setattr(args, "post_epochs", max_epochs)

    ## Train ##
    if args.tpu:
        # Two approaches to data loading MP and DP?
        # MP: https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet.py
        #train_loader = pl.MpDeviceLoader(train_loader, device)
        #test_loader = pl.MpDeviceLoader(test_loader, device)
        # DP: https://github.com/pytorch/xla/blob/master/test/test_train_imagenet.py
        #   Do nothing
        #
        # For torch_xla == 1.5
        train_kwargs = {
            "batch_size": train_loader.batch_size,
            "dataset_size": len(train_loader.dataset),
            "num_batches": len(train_loader)
        }
        #train_loader = pl.ParallelLoader(train_loader, [device])
        #train_loader = train_loader.per_device_loader(device)
        #test_loader = pl.ParallelLoader(test_loader, [device])
        #test_loader = test_loader.per_device_loader(device)

    print_fn("Training for {} epochs.".format(args.post_epochs))
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
        **train_kwargs
    )

    ## Display Results ##
    frames = [
        post_result.head(1),
        post_result.tail(1),
    ]
    train_result = pd.concat(frames, keys=["Init.", "Final"])
    print_fn("Train results:\n", train_result)

    ## Save Results and Model ##
    if args.save:
        print_fn("Saving results.")
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        torch.save(model.state_dict(), "{}/model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(), "{}/optimizer.pt".format(args.result_dir))
