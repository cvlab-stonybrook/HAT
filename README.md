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
#### Try out the [demo code](https://github.com/cvlab-stonybrook/HAT/blob/main/demo.ipynb) to generate a scanpath for your test image!
#### Commands
- Train a model with
    ```
    python train.py --hparams ./configs/coco_search18_dense_SSL.json --dataset-root <dataset_root> 
    ```

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
