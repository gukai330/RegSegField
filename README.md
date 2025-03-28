# RegSegField: Mask-Regularization and Hierarchical Segmentation for Novel View Synthesis from Sparse Inputs

## Overview
**RegSegField** is a novel pipeline that leverages 2D segmentations to aid the reconstruction of objects and parts for both novel view synthesis (NVS) and 3D scene segmentation under sparse-input conditions. It introduces a mask-visibility loss by matching 2D segments across different views, thereby defining distinct 3D regions for objects. Additionally, a hierarchical feature field supervised by contrastive learning enables iterative mask refinement, while a multi-level hierarchy loss addresses segmentation inconsistencies across views. Experiments show that this regularization approach outperforms various depth-guided NeRF methods and even enables sparse reconstruction of 3D-GS with random initialization.



## Installation
This repository is built upon [Nerfstudio 1.1.5](https://github.com/nerfstudio-project/nerfstudio) and gsplat 0.1.2.

### Prerequisites
- **Hardware**: NVIDIA GPU with CUDA installed (tested with CUDA 11.8). More details on installing CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
- **Python**: Version >= 3.8. It is recommended to use [Conda](https://docs.conda.io/miniconda.html) to manage dependencies.

### Environment Setup

1. **Create a Conda Environment**
    ```bash
    conda create --name regseg -y python=3.8
    conda activate regseg
    pip install --upgrade pip
    ```

2. **Install Dependencies**

    - **PyTorch with CUDA** (tested with CUDA 11.7 and 11.8):
      ```bash
      pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
      ```
    - **CUDA Toolkit**:
      ```bash
      conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
      ```
    - **Tiny CUDA NN**:
      ```bash
      pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
      ```

    For more details, see the [Dependencies section](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies) in the Nerfstudio installation documentation.

3. **Clone the Repository and Install Nerfstudio**
    ```bash
    git clone https://github.com/gukai330/RegSegField.git
    cd RegSegField
    pip install --upgrade pip setuptools
    pip install -e .
    ```

## Dataset Preparation
This work uses the **mip-NeRF 360** and **DTU** datasets. For the 2D segmentation part, [Semantic-SAM: Segment and Recognize Anything at Any Granularity](https://github.com/UX-Decoder/Semantic-SAM). is employed for image segmentation. Please configure its environment according to its GitHub project instructions, and place the `extract_mask.py` script (from the mask matching module) into its root directory.
Before running the mask matching or training steps

> 📍 **Note**: Please run this script from the root of the Semantic-SAM repository.

**Run for all scenes under a dataset folder:**

```bash
cd /path/to/Semantic-SAM

python extract_mask.py \
    --dataset_root /path/to/dataset \
    --sam_ckpt ./pretrained/swinl_only_sam_many2many.pth
```
### 2D Match Extraction with COLMAP

Before running the mask matching step, you must extract 2D keypoints and matches using COLMAP.  
To ensure fairness under sparse-input settings, we only extract 2D matches **from the training images**.  
Therefore, we re-run COLMAP using only the selected views to generate sparse 2D matches, and again on the full set to recover camera parameters and 3D points.

For DTU-style datasets:

```bash
python prepare_dtu_colmap.py \
    --raw_datadir /path/to/Rectified \
    --mask_dir /path/to/idr_masks \
    --output_dir /path/to/output_dataset \
    --light_condition 3
```

### Mask Matching

After extracting semantic segmentations using Semantic-SAM, run the **mask matching step** to train feature fields for consistent multi-view mask alignment.

Make sure your dataset directory has the following structure:


To run mask matching:

```bash
python mask_matching/mask_matching_main.py \
    --dataset_root /path/to/dtu_data \
    --binary_mask_path idr_masks \
    --recon_subdir sparse/0 \
    --input_views 3 \
    --downscales_factor 4 \
    --epochs 500 \
    --output_dir output_matching
```


## Training 

### Training on DTU and MIP-NeRF 360 Scenes 

To run RegSegField training experiments on multiple DTU and Mip-NeRF 360 scenes with different settings (e.g. with/without regularization and depth), use the provided script:

```bash
python train_dtu.py \
    --data_root /path/to/dtu_data \
    --scenes scan24 scan45 scan61 scan63 scan64 scan76 scan83 scan84 scan93 scan97

python train_mip360.py \
    --data_root /path/to/mip360 \
    --scenes counter kitchen garden bonsai room

```

## Acknowledgements

The selected scenes above are exactly those used in the results presented in our published paper.

Special thanks to the great work of the Nerfstudio comunity and the authors of Semantic SAM for making this possible.

If you find this repository useful, please consider citing our work in your publications.

```
@inproceedings{10.1145/3697294.3697299,
  author = {Gu, Kai and Maugey, Thomas and Sebastian, Knorr and Guillemot, Christine},
  title = {RegSegField: Mask-Regularization and Hierarchical Segmentation for Novel View Synthesis from Sparse Inputs},
  year = {2024},
  booktitle = {Proceedings of 21st ACM SIGGRAPH Conference on Visual Media Production (CVMP)}
}
```