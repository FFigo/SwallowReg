# SwallowReg

**SAM-Assisted Deformable Registration with Adaptive Global-Local Features for Cine-MRI Swallowing Function Quantification** (MICCAI 2026).

SwallowReg is an end-to-end joint segmentation–registration framework for tongue motion analysis in cine-MRI. It injects the semantic prior of the Segment Anything Model (SAM) into a deformable registration network through a multi-scale feature adaptor, and fuses SAM features with registration-encoder features via a Bidirectional Coordinate Attention Fusion (BCAF) module. An edge-weighted IoU loss further improves boundary alignment.

---

## Highlights

- **SAM Feature Adaptor** — a frozen SAM ViT-B image encoder combined with a learnable U-Net encoder produces 4-scale semantic features.
- **Swin-Intern Encoder** — Swin-Intern blocks run DCNv3 and window attention in parallel with learnable gating.
- **BCAF** — Bidirectional Coordinate Attention Fusion adaptively fuses SAM features and encoder features.
- **Edge-weighted IoU loss** — a Laplacian-weighted boundary loss for sharper anatomical alignment.
- A single entry point (`train_joint.py`) reproduces every ablation (Table 1) and baseline comparison (Table 2) via command-line flags.

---

## Repository structure

```
SwallowReg/
├── train_joint.py              # Training entry point
├── Infer.py                    # Testing / evaluation entry point
├── joint_model.py              # SwallowReg model (SAM adaptor + reg net + losses)
├── joint_Trainer.py            # Training loop
├── RegModel/
│   ├── reg_network.py          # SwinInternRegNet (ours): Swin-Intern encoder + BCAF
│   ├── VoxelMorph.py           # Baseline
│   └── TransMorph.py           # Baseline
├── SAMModel/
│   └── sam_feature_adaptor.py  # SAM Feature Adaptor
├── Utils/                      # Spatial transform, losses, dataset, decoder blocks
├── ops_dcnv3/                  # DCNv3 CUDA operator (must be compiled, see below)
├── SAMWeights/
│   └── down_weights.py         # Helper to download SAM pretrained weights
├── requirements.txt
└── README.md
```

---

## 1. Installation

Tested with **Python 3.12** and **PyTorch 2.7.1 (CUDA 11.8)** on an NVIDIA GPU.

```bash
# 1) Create environment
conda create -n swallowreg python=3.12 -y
conda activate swallowreg

# 2) Install PyTorch matching your CUDA version (example: CUDA 11.8)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# 3) Install the remaining Python dependencies
pip install -r requirements.txt

# 4) Install Segment Anything (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git

# 5) Compile the DCNv3 operator (requires a CUDA toolkit matching your PyTorch build)
cd ops_dcnv3
sh ./make.sh
cd ..
```

---

## 2. SAM pretrained weights

The SAM ViT-B checkpoint is **not** included in this repository. Download it into `SAMWeights/`:

```bash
python SAMWeights/down_weights.py
# downloads sam_vit_b_01ec64.pth (~358 MB) into SAMWeights/
```

The training script looks for `./SAMWeights/sam_vit_b_01ec64.pth` by default (override with `--sam_checkpoint`).

---

## 3. Dataset

The tongue cine-MRI dataset (coronal + sagittal) is hosted on Zenodo:

- **DOI:** https://doi.org/10.5281/zenodo.20724549

Download and unzip it into `Data/` so that the structure is:

```
Data/
├── Guan/Guan_data_npz/{train,val,test}/*.npz   # coronal view
└── Shi/Shi_data_npz/{train,val,test}/*.npz     # sagittal view
```

Each `.npz` holds a moving/fixed image pair and their segmentation masks.

The **ACDC** cardiac MRI dataset used in the paper is publicly available from its
[official source](https://www.creatis.insa-lyon.fr/Challenge/acdc/) and is not redistributed here.

---

## 4. Pretrained models

The trained SwallowReg models are released under
[**Releases**](https://github.com/FFigo/SwallowReg/releases):

| Model | View | Download |
|-------|------|----------|
| `swallowreg_guan_coronal.pth` | Coronal (Guan) | [link](https://github.com/FFigo/SwallowReg/releases/download/V1.0/swallowreg_guan_coronal.pth) |
| `swallowreg_shi_sagittal.pth` | Sagittal (Shi) | [link](https://github.com/FFigo/SwallowReg/releases/download/V1.0/swallowreg_shi_sagittal.pth) |

Place a model where `Infer.py` can find it (it auto-loads `best_model.pth` from the experiment
directory, and reads the configuration stored inside the checkpoint), e.g.:

```bash
mkdir -p Trained_models/SwallowReg/Guan_experiments/Full
# move the downloaded file and rename it to best_model.pth
mv swallowreg_guan_coronal.pth Trained_models/SwallowReg/Guan_experiments/Full/best_model.pth
```

---

## 5. Training

Basic usage (full model = our method with all components):

```bash
python train_joint.py \
    --data_path Data/Guan/Guan_data_npz \
    --save_dir experiments \
    --reg_net swin_intern \
    --fusion_mode full
```

Key arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Dataset root containing `train/`, `val/`, `test/` | (required) |
| `--save_dir` | Directory to save checkpoints / logs | (required) |
| `--sam_checkpoint` | SAM ViT-B weights | `./SAMWeights/sam_vit_b_01ec64.pth` |
| `--reg_net` | Registration network: `swin_intern` / `voxelmorph` / `transmorph` | `swin_intern` |
| `--fusion_mode` | Feature fusion: `full` / `concat` / `sam_only` / `enc_only` | `full` |
| `--use_edge_iou_loss` / `--no_edge_iou_loss` | Toggle the edge-weighted IoU loss | enabled |
| `--epochs`, `--batch_size`, `--learning_rate`, `--beta` | Training hyper-parameters | see `--help` |

If `--experiment_name` is not given, the output folder is auto-named
`{reg_net}_{fusion_mode}_{edge|noedge}_{timestamp}`, so different runs never overwrite each other.

---

## 6. Reproducing the ablation study (Table 1)

`SwallowRegNet` supports four feature-fusion modes, and the edge-weighted IoU loss is an
independent switch. Their combination reproduces all six rows of Table 1
(all on `--reg_net swin_intern`):

| Row | Configuration | Command |
|-----|---------------|---------|
| 1 | F_En only | `python train_joint.py --data_path <DATA> --save_dir experiments --fusion_mode enc_only --no_edge_iou_loss` |
| 2 | F_SAM only | `python train_joint.py --data_path <DATA> --save_dir experiments --fusion_mode sam_only --no_edge_iou_loss` |
| 3 | F_En + edge loss | `python train_joint.py --data_path <DATA> --save_dir experiments --fusion_mode enc_only` |
| 4 | F_SAM + edge loss | `python train_joint.py --data_path <DATA> --save_dir experiments --fusion_mode sam_only` |
| 5 | F_En + F_SAM concat (no BCAF) | `python train_joint.py --data_path <DATA> --save_dir experiments --fusion_mode concat` |
| 6 | Full (BCAF, our model) | `python train_joint.py --data_path <DATA> --save_dir experiments --fusion_mode full` |

- `enc_only`: registration-encoder features only.
- `sam_only`: SAM features only.
- `concat`: SAM + encoder features concatenated directly (no BCAF).
- `full`: BCAF bidirectional fusion (complete model).

---

## 7. Reproducing the baseline comparison (Table 2)

The same SAM scheme can be plugged into different registration backbones. `enc_only` is the
backbone without SAM; `full` is the backbone with the complete SAM scheme (SAM features + BCAF +
edge-weighted IoU loss).

| Method | Command |
|--------|---------|
| VoxelMorph (w/o SAM) | `python train_joint.py --data_path <DATA> --save_dir experiments --reg_net voxelmorph --fusion_mode enc_only` |
| VoxelMorph + SAM | `python train_joint.py --data_path <DATA> --save_dir experiments --reg_net voxelmorph --fusion_mode full` |
| TransMorph (w/o SAM) | `python train_joint.py --data_path <DATA> --save_dir experiments --reg_net transmorph --fusion_mode enc_only` |
| TransMorph + SAM | `python train_joint.py --data_path <DATA> --save_dir experiments --reg_net transmorph --fusion_mode full` |
| **Ours + SAM** | `python train_joint.py --data_path <DATA> --save_dir experiments --reg_net swin_intern --fusion_mode full` |

> Note: `voxelmorph` and `transmorph` only support `full` and `enc_only`. Passing `concat` or
> `sam_only` to a baseline raises a clear error.

---

## 8. Testing / evaluation

```bash
python Infer.py \
    --experiment_dir Trained_models/SwallowReg/Guan_experiments/Full \
    --test_data_dir Data/Guan/Guan_data_npz/test
```

- `--experiment_dir`: folder containing the model (`best_model.pth`). The configuration is read
  from `config.yaml` if present, otherwise from the checkpoint itself.
- `--checkpoint`: optionally point to a specific `.pth` file.
- `--output_dir`: optional; defaults to `Test_results/<experiment>` (kept separate from the models).

The script reports **Dice**, **GNCC**, **SSIM**, and the negative-Jacobian ratio, and saves
per-sample visualizations, deformation fields, and NIfTI outputs.

---

## Citation

If you find this work useful, please cite the paper (MICCAI 2026) and the dataset:

```
Yu, H., Jiang, C., Tang, Z., & Xia, C. (2026).
SwallowReg Tongue Cine-MRI Dataset (Coronal & Sagittal) [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.20724549
```

---

## Acknowledgements

This project builds on [Segment Anything](https://github.com/facebookresearch/segment-anything),
[InternImage / DCNv3](https://github.com/OpenGVLab/InternImage),
[VoxelMorph](https://github.com/voxelmorph/voxelmorph), and
[TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration).

## License

Code is released under the MIT License. The dataset is released under CC-BY-4.0 (see the Zenodo record).

## Contact

For questions, please contact: **1323048615@njupt.edu.cn**
