# train.py
import gzip
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from config import (
    USERNAME, TRAINING_DATA, MODEL_DIR,
    LEARNING_RATE, MAX_STEPS, EARLY_STOP_PATIENCE,
    MAIA_BASE_MODEL, FREEZE_RESIDUALS
)
from load_maia_weights import load_maia_weights

from move_mapping import POLICY_SIZE, get_gather_indices

BOARD_PLANES = 112 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture (Note for future me : It must perfectly match Maia / Leela Zero) 

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class MaiaNet(nn.Module):
    def __init__(self, num_blocks: int = 6, channels: int = 64) -> None:
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(112, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(channels, 80, kernel_size=3, padding=1, bias=True)
        # We use our shared gather_indices here!
        self.register_buffer("gather_indices", torch.tensor(get_gather_indices(), dtype=torch.long))
        
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1)
        )

    def forward(self, x):
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
            
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        policy_logits = p[:, self.gather_indices]
        
        value = self.value_head(x)
        return policy_logits, value

    def configure_freezing(self, freeze: bool) -> None:
        """
        Fixes the bug from v_1 where this actually unfroze the blocks.
        Now it properly respects the config setting.
        """
        for param in self.blocks.parameters():
            param.requires_grad = not freeze


# ── Dataset Loader ─────────────────────────────────────────────────────────────

class ChessPositionDataset(Dataset):
    def __init__(self, data_dir: Path) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        chunk_files = sorted(data_dir.glob("*.gz"))

        for path in chunk_files:
            with gzip.open(path, "rb") as f:
                planes = np.load(f)
                policy = np.load(f)
                winner = np.load(f)
            for i in range(len(planes)):
                self.samples.append((planes[i], policy[i], winner[i]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        planes, policy, winner = self.samples[idx]
        return (
            torch.from_numpy(planes),
            torch.from_numpy(policy),
            torch.tensor(winner, dtype=torch.float32),
        )

# ── Training Loop Utilities ────────────────────────────────────────────────────

def move_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    gold = targets.argmax(dim=1)
    return (pred == gold).float().mean().item()

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                step: int, max_steps: int) -> tuple[float, float, int]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for planes, policy, winner in loader:
        if step >= max_steps:
            break

        planes = planes.to(DEVICE)
        policy = policy.to(DEVICE)
        winner = winner.to(DEVICE)

        policy_logits, value_pred = model(planes)

        # Label smoothing spreads 10% of probability mass across all moves
        # This regularises the network and improves generalisation.
        policy_loss = F.cross_entropy(policy_logits, policy, label_smoothing=0.1)
        value_loss = F.mse_loss(value_pred.squeeze(), winner)
        loss = policy_loss + 0.01 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = move_accuracy(policy_logits, policy)
        total_loss += loss.item()
        total_acc  += acc
        n          += 1
        step       += 1

    return total_loss / max(n, 1), total_acc / max(n, 1), step

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for planes, policy, winner in loader:
        planes = planes.to(DEVICE)
        policy = policy.to(DEVICE)
        winner = winner.to(DEVICE)

        policy_logits, value_pred = model(planes)
        policy_loss = F.cross_entropy(policy_logits, policy, label_smoothing=0.1)
        value_loss  = F.mse_loss(value_pred.squeeze(), winner)
        loss        = policy_loss + 0.01 * value_loss

        total_loss += loss.item()
        total_acc  += move_accuracy(policy_logits, policy)
        n          += 1

    return total_loss / max(n, 1), total_acc / max(n, 1)

def main() -> None:
    print(f"\n=== Training personalized bot for {USERNAME} ===")
    print(f"    Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"    GPU    : {torch.cuda.get_device_name(0)}")

    train_dir = Path(TRAINING_DATA) / "train"
    val_dir   = Path(TRAINING_DATA) / "val"
    train_ds = ChessPositionDataset(train_dir)
    val_ds   = ChessPositionDataset(val_dir)

    print(f"\nDataset:\n  Train : {len(train_ds)} positions\n  Val   : {len(val_ds)} positions")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              num_workers=0)

    model = MaiaNet(num_blocks=6, channels=64).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel:\n  Parameters : {total_params:,}")

    maia_path = Path(MAIA_BASE_MODEL)
    if maia_path.exists():
        partial_state = load_maia_weights(str(maia_path), policy_size=POLICY_SIZE)
        missing, unexpected = model.load_state_dict(partial_state, strict=False)
        print(f"  Loaded pre-trained weights from {maia_path}")
    else:
        print(f"  ⚠ Maia weights not found at {maia_path} — Please copy them over from v_1/base_models!")
        return

    model.configure_freezing(FREEZE_RESIDUALS)
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params : {sum(p.numel() for p in trainable):,} ({FREEZE_RESIDUALS=})")

    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=LEARNING_RATE / 10)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc   = 0.0
    patience_count = 0
    step           = 0
    epoch          = 0

    print(f"\nTraining (max {MAX_STEPS} steps, early stop patience={EARLY_STOP_PATIENCE}):\n")
    print(f"  {'Epoch':>5}  {'Step':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Loss':>9}  {'Val Acc':>9}  {'Status':>10}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*10}")

    start_time = time.time()

    while step < MAX_STEPS:
        epoch += 1
        train_loss, train_acc, step = train_epoch(model, train_loader, optimizer, step, MAX_STEPS)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader)

        # Requires a 0.1-point improvement to reset patience
        if val_acc > best_val_acc + 0.001:   
            best_val_acc   = val_acc
            patience_count = 0
            status = "[OK] saved"
            torch.save({
                "epoch":       epoch,
                "step":        step,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "val_loss":    val_loss,
                "username":    USERNAME,
            }, model_dir / "best_model.pt")
        else:
            patience_count += 1
            status = f"patience {patience_count}/{EARLY_STOP_PATIENCE}"

        print(f"  {epoch:>5}  {step:>6}  {train_loss:>10.4f}  {train_acc*100:>9.2f}%  {val_loss:>9.4f}  {val_acc*100:>8.2f}%  {status}")

        if patience_count >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping - val accuracy hasn't improved for {EARLY_STOP_PATIENCE} epochs.")
            break

    total_time = time.time() - start_time
    print(f"\n-- Results -------------------------------------------")
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  Training time     : {total_time:.0f}s")
    print(f"  Model saved to    : {MODEL_DIR}/best_model.pt")

if __name__ == "__main__":
    main()
