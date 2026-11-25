# CNN vs. Transformer for image classification

CSCI-B 657 Computer Vision, Project 3. Meghana Nanuvala.

A CNN (ResNet-18) and a Transformer (a small Vision Transformer) are written in PyTorch,
trained from scratch on CIFAR-100 and compared in accuracy, efficiency and representation
ability. No pretrained weights are used. The report is `report.pdf`.

## What is in this repository

| File | What it is |
|---|---|
| `cnn.py` | the ResNet-18 model; running it trains the CNN |
| `vit.py` | the Vision Transformer model; running it trains the ViT |
| `train.py` | data loading, training loop and measurements, used by `cnn.py` and `vit.py` |
| `compare.py` | reads the results of both models, prints the comparison table and draws the figures |
| `architecture.png` | the architecture diagram shown in the report |
| `requirements.txt` | Python packages |
| `report.pdf` | the report |
| `results/` | output of the runs used in the report |

The dataset is not in the repository. The scripts download it on the first run.

## What you need

- Linux with an NVIDIA GPU. 2 GB of GPU memory is enough. Training does not run on a CPU.
- Python 3.12. This is the version I used, and the pinned packages in `requirements.txt`
  need Python 3.11 or 3.12.
- An internet connection for the first run, to download CIFAR-100 (169 MB, about 350 MB on
  disk after unpacking).

## Step 1: get the code

```bash
git clone https://github.com/meghanaNanuvala/cnn-vs-vit-cifar100.git
cd cnn-vs-vit-cifar100
```

Every command below is run from inside this folder.

## Step 2: install the packages

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Check that PyTorch sees the GPU. This must print `True`:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Step 3: run the experiment (about 25 minutes on one NVIDIA H100)

```bash
python cnn.py
python vit.py
python compare.py
```

`cnn.py` and `vit.py` can be run in either order. `compare.py` must come last, because it
reads the files the other two write.

The first run downloads CIFAR-100 into `data/`. Each training script prints one line per
epoch and a summary at the end. For `cnn.py` my run printed:

```
ResNet-18: 11.22M parameters, 0.56 GFLOPs
epoch   1/100  loss 3.914  train 9.67  test 15.24  4.9s
epoch   2/100  loss 3.252  train 19.89  test 23.30  4.8s
...
epoch 100/100  loss 0.010  train 99.95  test 77.38  4.1s
ResNet-18: test top-1 77.38  top-5 93.90  train top-1 99.97  k-NN 77.45
```

These commands write to `results/` and replace the files that came with the repository. To keep
those, add `--out my_results` to all three commands.

When the run is finished, the folder contains:

| File | Written by | Content |
|---|---|---|
| `cnn.json` | `cnn.py` | every number measured for the CNN, including the per-epoch history |
| `vit.json` | `vit.py` | the same for the ViT |
| `curves.png` | `compare.py` | training loss, training accuracy and test accuracy per epoch |
| `tsne.png` | `compare.py` | t-SNE of the features of 8 classes for both models |

The comparison table is printed on the screen. The `results/` folder in the repository also holds
what my run printed: `cnn_log.txt`, `vit_log.txt` and `comparison.txt`.

## Options

`cnn.py` and `vit.py` take the same options:

| Option | Default | Meaning |
|---|---|---|
| `--data` | `data` | folder for CIFAR-100 |
| `--out` | `results` | folder for the outputs |
| `--epochs` | 100 | training epochs |
| `--batch-size` | 128 | batch size |
| `--workers` | 8 | data loading processes |

`compare.py` only takes `--out`, the folder to read from and write to.

## What the scripts do

`cnn.py` and `vit.py` each build their model and then:

1. measure parameters, FLOPs and inference speed;
2. train the model and evaluate on the test set after every epoch;
3. record the training time per epoch, the peak GPU memory and the five classes with the lowest accuracy;
4. compute the features in front of the classifier for all images, classify the test images
   with a 20-nearest-neighbour vote on these features, and project the features of 8 classes
   to 2-D with t-SNE;
5. write everything to `cnn.json` or `vit.json` in the output folder.

`compare.py` reads both files, prints the comparison table and draws the two figures.

## Results I got

| | ResNet-18 | ViT |
|---|---|---|
| Test top-1 accuracy (%) | 77.38 | 58.82 |
| Test top-5 accuracy (%) | 93.90 | 82.56 |
| Training top-1 accuracy (%) | 99.97 | 99.97 |
| k-NN accuracy on features (%) | 77.45 | 57.91 |
| Parameters (M) | 11.22 | 10.73 |
| FLOPs per image (G) | 0.56 | 0.71 |
| Inference speed (images/s) | 103,925 | 65,851 |
| Training time per epoch (s) | 4.1 | 7.4 |
| Peak GPU memory in training (GB) | 0.54 | 1.23 |

The runs use seed 0, but GPU training is not fully deterministic, so a new run will not
reproduce the accuracies to the last digit. The speed numbers depend on the GPU.

## If something goes wrong

| Message or problem | What to do |
|---|---|
| `training needs a CUDA GPU` | PyTorch does not see a GPU. Run the check from step 2 on a machine with an NVIDIA GPU. |
| The CIFAR-100 download fails | The machine has no internet access. Copy `cifar-100-python.tar.gz` into `data/` by hand, or run `python cnn.py --epochs 1` once on a machine with a GPU and internet. |
| `results/cnn.json not found` from `compare.py` | Run `cnn.py` and `vit.py` first, with the same `--out` folder. |
| `pip` cannot find a pinned version | Your Python is not 3.11 or 3.12. Use one of these, or install the newest versions: `pip install torch torchvision fvcore scikit-learn matplotlib`. I did not test newer versions. |
| Warnings about too many data loading workers | Add `--workers 4`. |

## How I ran it on the IU research cluster

The cluster's Python module already has PyTorch, so only `fvcore` had to be installed.
Replace `<account>` and `<partition>` with your own.

```bash
module load python/gpu/3.12.5
python -m venv --system-site-packages venv
source venv/bin/activate
pip install fvcore

srun -A <account> -p <partition> --gres=gpu:1 --cpus-per-task=16 --mem=32G --time=01:00:00 \
     bash -c "python cnn.py && python vit.py && python compare.py"
```
