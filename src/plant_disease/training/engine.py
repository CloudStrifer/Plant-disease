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
