import torch


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(batch["image"].to(device))
        losses = loss_fn(outputs, {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()})
        losses["total"].backward()
        optimizer.step()
        running += float(losses["total"].detach().cpu())
    return running / max(len(loader), 1)


def evaluate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            losses = loss_fn(outputs, {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()})
            running += float(losses["total"].detach().cpu())
    return running / max(len(loader), 1)
