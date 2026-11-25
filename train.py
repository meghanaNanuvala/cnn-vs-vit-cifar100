"""Training and measurement code shared by cnn.py and vit.py."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_CLASSES = 100
device = torch.device("cuda")


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="where CIFAR-100 is stored (downloaded if missing)")
    parser.add_argument("--out", default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("training needs a CUDA GPU")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    return args


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def get_data(args):
    normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    plain = transforms.Compose([transforms.ToTensor(), normalize])

    train = datasets.CIFAR100(args.data, train=True, transform=augment, download=True)
    train_plain = datasets.CIFAR100(args.data, train=True, transform=plain)
    test = datasets.CIFAR100(args.data, train=False, transform=plain, download=True)

    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                  persistent_workers=True)
    loaders = {
        "train": DataLoader(train, shuffle=True, drop_last=True, **kwargs),
        "train_plain": DataLoader(train_plain, **kwargs),
        "test": DataLoader(test, **kwargs),
    }
    return loaders, test.classes


@torch.no_grad()
def predict(model, loader, features=False):
    """Logits for every image in the loader, or the feature vectors that feed the classifier."""
    model.eval()
    outputs, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with autocast():
            out = model.forward_features(images) if features else model(images)
        outputs.append(out.float().cpu())
        labels.append(targets)
    return torch.cat(outputs), torch.cat(labels)


def accuracy(logits, labels):
    top5 = logits.topk(5, dim=1).indices
    top1_acc = (top5[:, 0] == labels).float().mean().item() * 100
    top5_acc = (top5 == labels[:, None]).any(dim=1).float().mean().item() * 100
    return top1_acc, top5_acc


def measure_efficiency(model, batch_size=256):
    model.eval()
    flops = FlopCountAnalysis(model, torch.randn(1, 3, 32, 32, device=device))
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)

    batch = torch.randn(batch_size, 3, 32, 32, device=device)
    with torch.no_grad(), autocast():
        for _ in range(20):
            model(batch)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            model(batch)
        torch.cuda.synchronize()
        seconds = time.time() - start

    return {
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "flops_G": flops.total() / 1e9,
        "inference_img_per_s": 100 * batch_size / seconds,
    }


def fit(model, optimizer, loaders, epochs, warmup_epochs, grad_clip):
    steps_per_epoch = len(loaders["train"])
    total, warmup = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        start, loss_sum, correct = time.time(), 0.0, 0
        for images, targets in loaders["train"]:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast():
                logits = model(images)
            loss = F.cross_entropy(logits.float(), targets)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            correct += (logits.argmax(1) == targets).sum().item()
        seconds = time.time() - start

        top1, top5 = accuracy(*predict(model, loaders["test"]))
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / steps_per_epoch,
            "train_top1": 100 * correct / (steps_per_epoch * images.shape[0]),
            "test_top1": top1,
            "test_top5": top5,
            "seconds": seconds,
            "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9,
        })
        print(f"epoch {epoch:3d}/{epochs}  loss {history[-1]['train_loss']:.3f}  "
              f"train {history[-1]['train_top1']:.2f}  test {top1:.2f}  {seconds:.1f}s", flush=True)
    return history


def knn_accuracy(train_x, train_y, test_x, test_y, k=20):
    """Classify each test image by a vote of its k nearest training images in feature space."""
    train_x = F.normalize(train_x.to(device), dim=1)
    test_x = F.normalize(test_x.to(device), dim=1)
    nearest = (test_x @ train_x.T).topk(k, dim=1).indices
    votes = F.one_hot(train_y.to(device)[nearest], NUM_CLASSES).sum(dim=1)
    return (votes.argmax(1).cpu() == test_y).float().mean().item() * 100


def tsne_points(features, labels, classes, shown=8):
    """2-D t-SNE coordinates of the test features of the first few classes."""
    keep = labels < shown
    points = TSNE(n_components=2, init="pca", random_state=0).fit_transform(features[keep].numpy())
    return {"points": points.tolist(), "labels": labels[keep].tolist(), "classes": classes[:shown]}


def hardest_classes(logits, labels, classes, k=5):
    """The k classes with the lowest accuracy and the class each is most often mistaken for."""
    preds = logits.argmax(1)
    acc = torch.stack([(preds[labels == c] == c).float().mean() for c in range(NUM_CLASSES)]) * 100
    hardest = []
    for c in acc.argsort()[:k].tolist():
        wrong = preds[(labels == c) & (preds != c)]
        hardest.append({
            "class": classes[c],
            "accuracy": acc[c].item(),
            "most_confused_with": classes[wrong.bincount().argmax().item()],
        })
    return hardest


def run(key, name, model, optimizer, args, warmup_epochs=0, grad_clip=None):
    """Train one model and write everything measured about it to <out>/<key>.json."""
    loaders, classes = get_data(args)
    result = {"name": name, **measure_efficiency(model)}
    print(f"{name}: {result['params_M']:.2f}M parameters, {result['flops_G']:.2f} GFLOPs")

    history = fit(model, optimizer, loaders, args.epochs, warmup_epochs, grad_clip)
    result["train_s_per_epoch"] = sum(h["seconds"] for h in history) / len(history)
    result["peak_train_mem_GB"] = max(h["peak_mem_GB"] for h in history)

    logits, labels = predict(model, loaders["test"])
    result["top1"], result["top5"] = accuracy(logits, labels)
    result["train_top1"] = accuracy(*predict(model, loaders["train_plain"]))[0]
    result["hardest_classes"] = hardest_classes(logits, labels, classes)

    # Representation: how well do the features in front of the classifier
    # separate the classes on their own?
    train_x, train_y = predict(model, loaders["train_plain"], features=True)
    test_x, test_y = predict(model, loaders["test"], features=True)
    result["feature_size"] = train_x.shape[1]
    result["knn_top1"] = knn_accuracy(train_x, train_y, test_x, test_y)
    result["tsne"] = tsne_points(test_x, test_y, classes)
    result["history"] = history

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    with open(out / f"{key}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"{name}: test top-1 {result['top1']:.2f}  top-5 {result['top5']:.2f}  "
          f"train top-1 {result['train_top1']:.2f}  k-NN {result['knn_top1']:.2f}")
