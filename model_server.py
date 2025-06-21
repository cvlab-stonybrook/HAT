import os
from io import BytesIO

import numpy as np
import orjson
import torch
from flask import Flask, request
from PIL import Image
from torchvision import transforms

from common.config import JsonConfig
from common.utils import transform_fixations
from hat.models import HumanAttnTransformer

# Flask server
app = Flask("hat-model-server")
app.logger.setLevel("DEBUG")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Using device: {device}")

# Initialize model, config and load checkpoint
# Note: The model is initialized in freeview mode (Target present and absent modes are not supported)
hparams = JsonConfig("configs/coco_freeview_dense_SSL.json")
model = HumanAttnTransformer(
    hparams.Data,
    num_decoder_layers=hparams.Model.n_dec_layers,
    hidden_dim=hparams.Model.embedding_dim,
    nhead=hparams.Model.n_heads,
    ntask=1 if hparams.Data.TAP == "FV" else 18,
    tgt_vocab_size=hparams.Data.patch_count + len(hparams.Data.special_symbols),
    num_output_layers=hparams.Model.num_output_layers,
    separate_fix_arch=hparams.Model.separate_fix_arch,
    train_encoder=hparams.Train.train_backbone,
    train_pixel_decoder=hparams.Train.train_pixel_decoder,
    use_dino=hparams.Train.use_dino_pretrained_model,
    dropout=hparams.Train.dropout,
    dim_feedforward=hparams.Model.hidden_dim,
    parallel_arch=hparams.Model.parallel_arch,
    dorsal_source=hparams.Model.dorsal_source,
    num_encoder_layers=hparams.Model.n_enc_layers,
    output_centermap="centermap_pred" in hparams.Train.losses,
    output_saliency="saliency_pred" in hparams.Train.losses,
    output_target_map="target_map_pred" in hparams.Train.losses,
    transfer_learning_setting=hparams.Train.transfer_learn,
    project_queries=hparams.Train.project_queries,
    is_pretraining=False,
    output_feature_map_name=hparams.Model.output_feature_map_name,
).to(device)

# The model weights are modified below due to changes in Detectron2
checkpoint_path = "./checkpoints/HAT_FV.pt"
ckpt = torch.load(checkpoint_path, map_location="cpu")
ckpt_new = ckpt.copy()
for k, v in list(ckpt["model"].items()):
    if "stages." in k:
        ckpt_new["model"][k.replace("stages.", "")] = v
        ckpt_new["model"].pop(k)
model.load_state_dict(ckpt_new["model"])
model.eval()

# Image Transform
transform = transforms.Compose(
    [
        transforms.Resize(size=(hparams.Data.im_h, hparams.Data.im_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ]
)


def get_fixation_history(x_hist, y_hist, image_size):
    # Fixations are normalized and truncated to max_traj_length (20 in case of freeview)
    x_hist = x_hist / image_size[0]
    y_hist = y_hist / image_size[1]
    fixation_hist = np.stack([x_hist, y_hist], axis=1)
    fixation_hist = np.expand_dims(fixation_hist, axis=0)
    fixation_hist = fixation_hist[:, -hparams.Data.max_traj_length :, :]
    return fixation_hist


@app.route("/conditional_log_density", methods=["POST"])
def conditional_log_density():
    # Extract stimulus
    image_bytes = request.files["stimulus"].read()
    image = Image.open(BytesIO(image_bytes))

    # Extract scanpath history
    data = orjson.loads(request.form["json_data"])
    x_hist = np.array(data["x_hist"])
    y_hist = np.array(data["y_hist"])
    fixation_hist = get_fixation_history(x_hist, y_hist, image.size)

    # Make tensors for model
    # Image is resized to (320, 512) for compatibility with the model
    image_tensor = transform(image).to(device)
    normalized_fixs = torch.tensor(fixation_hist)
    task_ids = torch.tensor(
        [0], device=device
    )  # task_ids are not used in freeview mode
    ys, ys_high = transform_fixations(
        normalized_fixs, None, hparams.Data, False, return_highres=True
    )  # transform fixations to categorical labels
    ys = ys.to(device)
    ys_high = ys_high.to(device)
    padding = None

    with torch.no_grad():
        # Extract image features
        dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps = model.encode(
            image_tensor
        )

        # Predict fixation heatmap
        out = model.decode_and_predict(
            dorsal_embs.clone(),
            dorsal_pos,
            dorsal_mask,
            high_res_featmaps,
            ys,
            padding,
            ys_high,
            task_ids,
        )
        heatmap = out["pred_fixation_map"]  # prob from (0, 1) for each pixel

        # Create conditional log density from heatmap
        heatmap = torch.nn.functional.interpolate(
            heatmap.view(1, 1, hparams.Data.im_h, hparams.Data.im_w),  # (1, 1, H, W)
            size=(image.size[1], image.size[0]),  # (height, width)
            mode="bicubic",
        )
        # bicubic interpolation can create negative value; clamp to epsilon for stability
        heatmap = heatmap.clamp(min=1e-30)
        prob = heatmap.view(image.size[1], image.size[0]) / heatmap.sum()
        log_density = prob.log()

    log_density_list = log_density.cpu().tolist()
    response = orjson.dumps({"log_density": log_density_list})
    return response


@app.route("/type", methods=["GET"])
def type():
    type = "HumanAttnTransformer"
    version = "v1.0.0"
    return orjson.dumps({"type": type, "version": version})


def main():
    app.run(host="localhost", port="4000", debug="True", threaded=True)


if __name__ == "__main__":
    main()
