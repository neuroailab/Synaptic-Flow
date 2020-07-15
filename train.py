import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


def train(
    model,
    loss,
    optimizer,
    dataloader,
    device,
    epoch,
    verbose,
    log_interval=10,
    save_freq=100,
    save_path=None,
):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        curr_step = epoch * len(dataloader) + batch_idx
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(dataloader.dataset),
                    100.0 * batch_idx / len(dataloader),
                    train_loss.item(),
                )
            )
        # TODO: this is just to be able to save at any step (even mid-epoch)
        #       it might make more sense to checkpoint only on epoch: makes
        #       for a cleaner codebase and can include test metrics
        # TODO: additionally, could integrate tfutils.DBInterface here
        if save_path is not None and curr_step % save_freq == 0:
            print("Saving model checkpoint")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss.item(),
                },
                f"{save_path}_ckpt_step{curr_step}.tar",
            )
    return total / len(dataloader.dataset)


def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:, :1].sum().item()
            correct5 += correct[:, :5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100.0 * correct1 / len(dataloader.dataset)
    accuracy5 = 100.0 * correct5 / len(dataloader.dataset)
    if verbose:
        print(
            "Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)".format(
                average_loss, correct1, len(dataloader.dataset), accuracy1
            )
        )
    return average_loss, accuracy1, accuracy5


def train_eval_loop(
    model,
    loss,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    device,
    epochs,
    verbose,
    save_freq=100,
    save_path=None,
):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        train_loss = train(
            model,
            loss,
            optimizer,
            train_loader,
            device,
            epoch,
            verbose,
            save_freq=save_freq,
            save_path=save_path,
        )
        test_loss, accuracy1, accuracy5 = eval(
            model, loss, test_loader, device, verbose
        )
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
    columns = ["train_loss", "test_loss", "top1_accuracy", "top5_accuracy"]
    return pd.DataFrame(rows, columns=columns)
