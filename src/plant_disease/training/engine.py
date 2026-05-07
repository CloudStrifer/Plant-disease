import torch


def train_one_epoch(model, loader, optimizer, loss_fn, device, log_interval: int | None = None, log_prefix: str = "train"):
    model.train()
    running = 0.0
    total_steps = len(loader)
    for step_index, batch in enumerate(loader, start=1):
        optimizer.zero_grad()
        outputs = model(batch["image"].to(device))
        losses = loss_fn(outputs, {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()})
        losses["total"].backward()
        optimizer.step()
        step_loss = float(losses["total"].detach().cpu())
        running += step_loss
        if log_interval and (step_index == 1 or step_index % log_interval == 0 or step_index == total_steps):
            print(f"{log_prefix} step={step_index}/{total_steps} loss={step_loss:.4f}", flush=True)
    return running / max(total_steps, 1)


def evaluate_one_epoch(model, loader, loss_fn, device, log_interval: int | None = None, log_prefix: str = "val"):
    model.eval()
    running = 0.0
    total_steps = len(loader)
    with torch.no_grad():
        for step_index, batch in enumerate(loader, start=1):
            outputs = model(batch["image"].to(device))
            losses = loss_fn(outputs, {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()})
            step_loss = float(losses["total"].detach().cpu())
            running += step_loss
            if log_interval and (step_index == 1 or step_index % log_interval == 0 or step_index == total_steps):
                print(f"{log_prefix} step={step_index}/{total_steps} loss={step_loss:.4f}", flush=True)
    return running / max(total_steps, 1)
