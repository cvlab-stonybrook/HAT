# HAT
CVPR 2024 "Unifying Top-down and Bottom-up Scanpath Prediction Using Transformers"

#### Installation
- Install customized Detectron2: Copy the folder at `/data/add_disk0/zbyang/projects/detectron2/` in `bigrod` to your working directory and install it as follows 
    ```
    cd detectron2
    pip install -e .
    ```
 - Install MSDeformableAttn:
   ```
   cd ./scanpath_prediction_all/sptransformer/pixel_decoder/ops
   sh make.sh
   ```

#### Commands
- Train a model with
    ```
    python train_sptransformer.py --hparams ./configs/coco_search18_dense_SSL.json --dataset-root <dataset_root> 
    ```

## Reference
This repository contains code for scanpath prediction models for the following papers. Please cite if you use this code base.

```bibtex
@InProceedings{yang2023unify,
  author = {Yang, Zhibo and Mondal, Sounak and Ahn, Seoyoung and Xue, Ruoyu and Zelinsky, Gregory and Hoai, Minh and Samaras, Dimitris},
  title = {Unifying Top-down and Bottom-up Scanpath Prediction Using Transformers},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024}
}

```
