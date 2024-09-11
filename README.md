# HAT
CVPR 2024 "Unifying Top-down and Bottom-up Scanpath Prediction Using Transformers"

#### Installation
 - Install [Detectron2](https://github.com/facebookresearch/detectron2)
 - Install MSDeformableAttn:
   ```
   cd ./hat/pixel_decoder/ops
   sh make.sh
   ```
 - Download pretrained model weights (ResNet-50 and Deformable Transformer) with the following python code
   ```
    if not os.path.exists("./pretrained_models/"):
        os.mkdir('./pretrained_models')

    print('downloading pretrained model weights...')
    url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl"
    wget.download(url, 'pretrained_models/')
    url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50.pkl"
    wget.download(url, 'pretrained_models/')
   ```
- Prepare the data following https://github.com/cvlab-stonybrook/Scanpath_Prediction.
- Download data from https://drive.google.com/drive/folders/1GOmWlDpG6Lh7iPlt9Hi9GWZ4BAfwhmHR?usp=drive_link.
#### Try out the [demo code](https://github.com/cvlab-stonybrook/HAT/blob/main/demo.ipynb) to generate a scanpath for your test image!
#### Commands
- Train a model with
    ```
    python train.py --hparams ./configs/coco_search18_dense_SSL.json --dataset-root <dataset_root> 
    ```
- Steps to train HAT on your custom dataset:
1.	Modify the configuration file: Update the values for Data.name, Data.TAP, and Data.max_traj_length in the config file to match your dataset's specifications.
2.	Create a fixation file: Generate a fixation.json file in the same format as coco_freeview_fixations_all.json. If your dataset doesn't have specific image categories, you can set the "task" field to "none".
3.	Load your dataset: Refer to the implementations for loading OSIE and MIT1003 in hat/builder.py, common/dataset.py, and common/data.py as a guide to integrate your own dataset.
4.	Sequence Score calculation: We will soon release the code for computing cluster.npy, required for calculating the Sequence Score. Currently, Sequence Score and Semantic Sequence Score are only supported for COCO-Search18 and COCO-Freeview datasets. For now, skip this calculation during evaluation.


## Reference
This repository contains code for scanpath prediction models for the following papers. Please cite if you use this code base.

```bibtex
@InProceedings{yang2024unify,
  author = {Yang, Zhibo and Mondal, Sounak and Ahn, Seoyoung and Xue, Ruoyu and Zelinsky, Gregory and Hoai, Minh and Samaras, Dimitris},
  title = {Unifying Top-down and Bottom-up Scanpath Prediction Using Transformers},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024}
}

```
