import torch
import torch.nn as nn
from os.path import join, dirname
from typing import Optional
from torch import Tensor
import common.position_encoding as pe
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from .backbone.swin import D2SwinTransformer
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .pixel_decoder.fpn import TransformerEncoderPixelDecoder
from .transformer_decoder.transformer import TransformerEncoder, TransformerEncoderLayer
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from .config import add_maskformer2_config
import fvcore.nn.weight_init as weight_init


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long())  # * math.sqrt(self.emb_size)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self,
                    tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self,
                tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask,
                                    query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask,
                                 query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(self,
                    tgt,
                    memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self,
                tgt,
                memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(self,
                 d_model,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class ImageFeatureEncoder(nn.Module):
    def __init__(self,
                 cfg_path,
                 dropout,
                 pixel_decoder='MSD',
                 load_segm_decoder=False,
                 pred_saliency=False):
        super(ImageFeatureEncoder, self).__init__()

        # Load Detectrion2 backbone
        cfg = get_cfg()
        add_maskformer2_config(cfg)
        cfg.merge_from_file(cfg_path)
        self.backbone = build_backbone(cfg)
        # if os.path.exists(cfg.MODEL.WEIGHTS):
        bb_weights = torch.load(cfg.MODEL.WEIGHTS,
                                map_location=torch.device('cpu'))
        bb_weights_new = bb_weights.copy()
        for k, v in bb_weights.items():
            if 'stages.' in k:
                bb_weights_new[k.replace('stages.', '')] = v
                bb_weights_new.pop(k)
        self.backbone.load_state_dict(bb_weights_new)
        self.backbone.eval()
        print('Loaded backbone weights from {}'.format(cfg.MODEL.WEIGHTS))

        if pred_saliency:
            assert not load_segm_decoder, "cannot load segmentation decoder and predict saliency at the same time"
            self.saliency_head = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, padding=0))
        else:
            self.saliency_head = None

        # Load deformable pixel decoder
        if cfg.MODEL.BACKBONE.NAME == 'D2SwinTransformer':
            input_shape = {
                "res2": ShapeSpec(channels=128, stride=4),
                "res3": ShapeSpec(channels=256, stride=8),
                "res4": ShapeSpec(channels=512, stride=16),
                "res5": ShapeSpec(channels=1024, stride=32)
            }
        else:
            input_shape = {
                "res2": ShapeSpec(channels=256, stride=4),
                "res3": ShapeSpec(channels=512, stride=8),
                "res4": ShapeSpec(channels=1024, stride=16),
                "res5": ShapeSpec(channels=2048, stride=32)
            }
        args = {
            'input_shape': input_shape,
            'conv_dim': 256,
            'mask_dim': 256,
            'norm': 'GN',
            'transformer_dropout': dropout,
            'transformer_nheads': 8,
            'transformer_dim_feedforward': 1024,
            'transformer_enc_layers': 6,
            'transformer_in_features': ['res3', 'res4', 'res5'],
            'common_stride': 4,
        }
        if pixel_decoder == 'MSD':
            msd = MSDeformAttnPixelDecoder(**args)
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + '_MSDeformAttnPixelDecoder.pkl'
            # if os.path.exists(ckpt_path):
            msd_weights = torch.load(ckpt_path,
                                     map_location=torch.device('cpu'))
            msd_weights_new = msd_weights.copy()
            for k, v in msd_weights.items():
                if k[:7] == 'adapter':
                    msd_weights_new["lateral_convs." + k] = v
                    msd_weights_new.pop(k)
                elif k[:5] == 'layer':
                    msd_weights_new["output_convs." + k] = v
                    msd_weights_new.pop(k)
            msd.load_state_dict(msd_weights_new)
            print('Loaded MSD pixel decoder weights from {}'.format(ckpt_path))
            self.pixel_decoder = msd
            self.pixel_decoder.eval()
        elif pixel_decoder == 'FPN':
            args.pop('transformer_in_features')
            args.pop('common_stride')
            args['transformer_dim_feedforward'] = 2048
            args['transformer_pre_norm'] = False
            fpn = TransformerEncoderPixelDecoder(**args)
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + '_FPN.pkl'
            # if os.path.exists(ckpt_path):
            fpn_weights = torch.load(ckpt_path,
                                     map_location=torch.device('cpu'))
            fpn.load_state_dict(fpn_weights)
            self.pixel_decoder = fpn
            print('Loaded FPN pixel decoder weights from {}'.format(ckpt_path))
            self.pixel_decoder.eval()
        else:
            raise NotImplementedError

        # Load segmentation decoder
        self.load_segm_decoder = load_segm_decoder
        if self.load_segm_decoder:
            args = {
                "in_channels": 256,
                "mask_classification": True,
                "num_classes": 133,
                "hidden_dim": 256,
                "num_queries": 100,
                "nheads": 8,
                "dim_feedforward": 2048,
                "dec_layers": 9,
                "pre_norm": False,
                "mask_dim": 256,
                "enforce_input_project": False,
            }
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + '_transformer_decoder.pkl'
            mtd = MultiScaleMaskedTransformerDecoder(**args)
            mtd_weights = torch.load(ckpt_path,
                                     map_location=torch.device('cpu'))
            mtd.load_state_dict(mtd_weights)
            self.segm_decoder = mtd
            print('Loaded segmentation decoder weights from {}'.format(
                ckpt_path))
            self.segm_decoder.eval()

    def forward(self, x):
        features = self.backbone(x)
        high_res_featmaps, _, ms_feats = \
            self.pixel_decoder.forward_features(features)
        if self.load_segm_decoder:
            segm_predictions = self.segm_decoder.forward(
                ms_feats, high_res_featmaps)
            queries = segm_predictions["out_queries"]

            segm_results = self.segmentation_inference(segm_predictions)
            # segm_results = None
            return high_res_featmaps, queries, segm_results
        else:
            if self.saliency_head is not None:
                saliency_map = self.saliency_head(high_res_featmaps)
                return {'pred_saliency': saliency_map}
            else:
                return high_res_featmaps, ms_feats[0], ms_feats[1]

    def segmentation_inference(self, segm_preds):
        """Compute panoptic segmentation from the outputs of the segmentation decoder."""
        mask_cls_results = segm_preds.pop("pred_logits")
        mask_pred_results = segm_preds.pop("pred_masks")

        processed_results = []
        for mask_cls_result, mask_pred_result in zip(mask_cls_results,
                                                     mask_pred_results):
            panoptic_r = self.panoptic_inference(mask_cls_result,
                                                 mask_pred_result)
            processed_results.append(panoptic_r)

        return processed_results

    def panoptic_inference(self,
                           mask_cls,
                           mask_pred,
                           object_mask_threshold=0.8,
                           overlap_threshold=0.8):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        # Remove non-object masks and masks with low confidence
        keep = labels.ne(mask_cls.size(-1) -
                         1) & (scores > object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w),
                                   dtype=torch.int32,
                                   device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        keep_ids = torch.where(keep)[0]

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return [], [], keep
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in range(80)
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item(
                ) > 0:
                    if mask_area / original_area < overlap_threshold:
                        keep[keep_ids[k]] = False
                        continue

                    # Commented out to keep indiviual stuff masks
                    # # merge stuff regions
                    # if not isthing:
                    #     if int(pred_class) in stuff_memory_list.keys():
                    #         panoptic_seg[mask] = stuff_memory_list[int(
                    #             pred_class)]
                    #         # continue
                    #     else:
                    #         stuff_memory_list[int(
                    #             pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    my, mx = torch.where(mask)
                    segments_info.append({
                        "id":
                        current_segment_id,
                        "isthing":
                        bool(isthing),
                        "category_id":
                        int(pred_class),
                        "mask_area":
                        mask_area,
                        "mask_centroid": (mx.float().mean(), my.float().mean())
                    })
                else:
                    keep[keep_ids[k]] = False

            return panoptic_seg, segments_info, keep


# Dense prediction transformer
class HumanAttnTransformer(nn.Module):
    def __init__(
        self,
        pa,
        num_decoder_layers: int,
        hidden_dim: int,
        nhead: int,
        ntask: int,
        tgt_vocab_size: int,
        num_output_layers: int,
        separate_fix_arch: bool = False,
        train_encoder: bool = False,
        train_foveator: bool = True,
        train_pixel_decoder: bool = False,
        use_dino: bool = True,
        pre_norm: bool = False,
        dropout: float = 0.1,
        dim_feedforward: int = 512,
        parallel_arch: bool = False,
        dorsal_source: list = ["P2"],
        num_encoder_layers: int = 3,
        output_centermap: bool = False,
        output_saliency: bool = False,
        output_target_map: bool = False,
        combine_pos_emb: bool = True,
        combine_all_emb: bool = False,
        transfer_learning_setting: str = 'none',
        project_queries: bool = True,
        is_pretraining: bool = False,
        output_feature_map_name: str = 'P4',
    ):
        super(HumanAttnTransformer, self).__init__()
        self.pa = pa
        self.num_decoder_layers = num_decoder_layers
        self.is_pretraining = is_pretraining
        self.combine_pos_emb = combine_pos_emb
        self.combine_all_emb = combine_all_emb
        self.output_feature_map_name = output_feature_map_name
        self.parallel_arch = parallel_arch
        self.dorsal_source = dorsal_source
        assert len(dorsal_source) > 0, "need to specify dorsal source: P1, P2!"
        self.output_centermap = output_centermap
        self.output_saliency = output_saliency
        self.output_target_map = output_target_map

        # Encoder: Deformable Attention Transformer
        self.train_encoder = train_encoder
        self.encoder = ImageFeatureEncoder(pa.backbone_config, dropout,
                                           pa.pixel_decoder)
        self.symbol_offset = len(self.pa.special_symbols)
        if not train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.train_pixel_decoder = train_pixel_decoder
        if train_pixel_decoder:
            self.encoder.pixel_decoder.train()
            for param in self.encoder.pixel_decoder.parameters():
                param.requires_grad = True
        featmap_channels = 256
        if hidden_dim != featmap_channels:
            self.input_proj = nn.Conv2d(featmap_channels,
                                         hidden_dim,
                                         kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()

        # Transfer learning setting (only support COCO-Search18 for now, where
        # we assume 18 search targets).
        self.transfer_learning_setting = transfer_learning_setting
        self.project_queries = project_queries
        if transfer_learning_setting == 'freeview2search':
            if project_queries:
                assert ntask == 18, "only support 18 task for freeview2search"
                # TODO: Create a new query for every new search task
                ntask = 1
                self.new_query_embed = nn.Embedding(18, hidden_dim)
                self.freeview2search_mlp = MLP(1, 64, 18, 3)
        elif transfer_learning_setting == 'search2freeview':
            if project_queries:
                assert ntask == 1, "only support 1 task for search2freeview"
                # Create a new query for the new free-viewing task
                ntask = 18
                self.new_query_embed = nn.Embedding(1, hidden_dim)
                self.search2freeview_mlp = MLP(18, 64, 1, 3)
        elif transfer_learning_setting == 'none' or transfer_learning_setting == 'finetune':
            pass
        else:
            raise ValueError(
                f"transfer_learning_setting {transfer_learning_setting} not supported."
            )

        # Queries
        self.ntask = ntask
        self.aux_queries = 0
        if self.output_centermap:
            self.aux_queries += 80
        if self.output_saliency:
            self.aux_queries += 1
        self.query_embed = nn.Embedding(ntask + self.aux_queries, hidden_dim)
        self.query_pos = nn.Embedding(ntask + self.aux_queries, hidden_dim)

        # Decoder
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_dorsal = nn.ModuleList()
        self.transformer_cross_attention_layers_ventral = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            self.transformer_cross_attention_layers_dorsal.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            if not self.parallel_arch:
                self.transformer_cross_attention_layers_ventral.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    ))
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))

        self.num_encoder_layers = num_encoder_layers
        if self.parallel_arch and num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=pre_norm)
            encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
            self.working_memory_encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)
            if not train_foveator:
                self.working_memory_encoder.eval()
                for param in self.working_memory_encoder.parameters():
                    param.requires_grad = False

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Task heads
        self.termination_predictor = MLP(hidden_dim + 1, hidden_dim, 1,
                                         num_output_layers)
        self.fixation_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                  num_output_layers)
        if self.output_target_map:
            self.target_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                    num_output_layers)
        if self.output_centermap:
            self.centermap_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                       num_output_layers)
        if self.output_saliency:
            self.saliency_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                      num_output_layers)

        # Positional embedding
        self.pixel_loc_emb = pe.PositionalEncoding2D(pa,
                                                     hidden_dim,
                                                     height=pa.im_h // 4,
                                                     width=pa.im_w // 4,
                                                     dropout=dropout)
        if self.output_feature_map_name == 'P4':
            self.pos_scale = 1
        elif self.output_feature_map_name == 'P2':
            self.pos_scale = 4
        else:
            raise NotImplementedError
        # self.positional_encoding = pe.PositionalEncoding(hidden_dim)

        self.fix_ind_emb = nn.Embedding(pa.max_traj_length, hidden_dim)

        # Embedding for distinguishing dorsal or ventral embeddings
        self.dorsal_ind_emb = nn.Embedding(2, hidden_dim)  # P1 and P2
        self.ventral_ind_emb = nn.Embedding(1, hidden_dim)

        # self.visual_pos_fuse_layer = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, padding=0)
        if self.is_pretraining:
            self.pretrain_task_head = MLP(hidden_dim, 32, 1, 3)

    def forward(self,
                img: torch.Tensor,
                tgt_seq: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                tgt_seq_high: torch.Tensor,
                task_ids: torch.Tensor = None,
                return_attn_weights=False,
                img_ids: torch.Tensor = None):

        # Prepare dorsal embeddings
        img_embs_s4, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(img_embs_s4)
        if self.output_feature_map_name == 'P4':
            output_featmaps = high_res_featmaps
        elif self.output_feature_map_name == 'P2':
            output_featmaps = self.input_proj(img_embs_s2)
        else:
            raise NotImplementedError

        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        if "P1" in self.dorsal_source:
            # C x 10 x 16
            img_embs = self.input_proj(img_embs_s1)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=8)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[0].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        if "P2" in self.dorsal_source:
            # C x 20 x 32
            img_embs = self.input_proj(img_embs_s2)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=4)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[1].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))

        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)

        if img_ids is not None and self.is_pretraining:
            # Pre-training: combine all images with every scanpath and
            # label each image-scanpath pair with image name index
            bs = img.shape[0]
            high_res_featmaps = high_res_featmaps.unsqueeze(0).expand(bs, bs, *high_res_featmaps.shape[1:])
            high_res_featmaps = high_res_featmaps.reshape(bs * bs, *high_res_featmaps.shape[2:])
            l, bs, ch = dorsal_embs.shape
            dorsal_embs = dorsal_embs.unsqueeze(1).expand(l, bs, bs, ch).reshape(l, bs * bs, ch)
            dorsal_pos = dorsal_pos.unsqueeze(1).expand(l, bs, bs, ch).reshape(l, bs * bs, ch)
            scale_embs =  scale_embs.unsqueeze(1).expand(l, bs, bs, ch).reshape(l, bs * bs, ch)
            tgt_len = tgt_seq_high.shape[1]
            tgt_seq = tgt_seq.unsqueeze(1).expand(bs, bs, tgt_len).reshape(bs * bs, tgt_len)
            tgt_seq_high = tgt_seq_high.unsqueeze(1).expand(bs, bs, tgt_len).reshape(bs * bs, tgt_len)
            tgt_padding_mask = tgt_padding_mask.unsqueeze(1).expand(bs, bs, tgt_len).reshape(bs * bs, tgt_len)

        bs = high_res_featmaps.size(0)
        # Prepare ventral embeddings
        if tgt_seq_high is None:
            tgt_seq = tgt_seq.transpose(0, 1)
            ventral_embs = torch.gather(
                torch.cat([
                    torch.zeros(1, *img_embs.shape[1:],
                                device=img_embs.device), img_embs
                ],
                          dim=0), 0,
                tgt_seq.unsqueeze(-1).expand(*tgt_seq.shape,
                                             img_embs.size(-1)))
            ventral_pos = self.pixel_loc_emb(
                tgt_seq)  # Pos for fixation location
        else:
            tgt_seq_high = tgt_seq_high.transpose(0, 1)
            highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
            ventral_embs = torch.gather(
                torch.cat([
                    torch.zeros(1,
                                *highres_embs.shape[1:],
                                device=img_embs.device), highres_embs
                ],
                          dim=0), 0,
                tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape,
                                                  highres_embs.size(-1)))
            # Pos for fixation location
            ventral_pos = self.pixel_loc_emb(tgt_seq_high)

        # Add pos into embeddings for attention prediction
        if self.combine_pos_emb:
            # Dorsal embeddings
            dorsal_embs += dorsal_pos
            dorsal_pos.fill_(0)
            # Ventral embeddings
            ventral_embs += ventral_pos
            ventral_pos.fill_(0)
            # High-res featmap
            # high_res_featmaps = self.visual_pos_fuse_layer(
            #     torch.cat([high_res_featmaps,
            #                self.pixel_loc_emb.forward_featmaps(
            #                    high_res_featmaps.shape[-2:]).expand_as(high_res_featmaps)], dim=-3))
            output_featmaps += self.pixel_loc_emb.forward_featmaps(
                output_featmaps.shape[-2:], scale=self.pos_scale)

        # Add embedding indicator embedding into pos embedding
        dorsal_pos += scale_embs
        ventral_pos += self.ventral_ind_emb.weight.unsqueeze(0).expand(
            *ventral_pos.shape)

        # Temporal embedding for fixations
        # ventral_pos += self.positional_encoding(
        #     ventral_embs).repeat(1, bs, 1)
        ventral_pos += self.fix_ind_emb.weight[:ventral_embs.
                                               size(0)].unsqueeze(1).repeat(
                                                   1, bs, 1)
        ventral_pos[tgt_padding_mask.transpose(0, 1)] = 0

        if self.combine_all_emb:
            dorsal_embs += dorsal_pos
            ventral_embs += ventral_pos
            dorsal_pos = ventral_pos = None

        # Update working memory
        if self.parallel_arch:
            working_memory = torch.cat([dorsal_embs, ventral_embs], dim=0)
            padding_mask = torch.cat([
                torch.zeros(bs, dorsal_embs.size(0),
                            device=dorsal_embs.device).bool(), tgt_padding_mask
            ],
                                     dim=1)
            if self.combine_all_emb:
                working_memory_pos = None
            else:
                working_memory_pos = torch.cat([dorsal_pos, ventral_pos],
                                               dim=0)
            # working_memory = dorsal_embs # ventral_embs
            # padding_mask = None # tgt_padding_mask
            # working_memory_pos = dorsal_pos # ventral_pos

            if self.num_encoder_layers > 0:
                working_memory = self.working_memory_encoder(
                    working_memory,
                    src_key_padding_mask=padding_mask,
                    pos=working_memory_pos)
            dorsal_embs = working_memory
            dorsal_pos = working_memory_pos
        else:
            padding_mask = None

        # Update queries with attention
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pos = None

        # Let model know which fixation it's going to predict
        num_fixs = (tgt_padding_mask.size(1) -
                    torch.sum(tgt_padding_mask, dim=1)).unsqueeze(0).expand(
                        self.ntask, bs)
        # num_fix_embed = self.fix_ind_emb.weight[num_fixs - 1]
        # query_embed += num_fix_embed

        attn_weights_all_layers = []
        for i in range(self.num_decoder_layers):
            # Dorsal cross attention
            query_embed, attn_weights = self.transformer_cross_attention_layers_dorsal[
                i](query_embed,
                   dorsal_embs,
                   memory_mask=None,
                   memory_key_padding_mask=padding_mask,
                   pos=dorsal_pos,
                   query_pos=query_pos)
            if return_attn_weights:
                attn_weights_all_layers.append(attn_weights)

            if not self.parallel_arch:
                # Ventral cross attention
                query_embed, _ = self.transformer_cross_attention_layers_ventral[
                    i](query_embed,
                       ventral_embs,
                       memory_mask=None,
                       memory_key_padding_mask=tgt_padding_mask,
                       pos=ventral_pos,
                       query_pos=query_pos)

            # if self.ntask > 1:
            #     # Self attention
            #     query_embed = self.transformer_self_attention_layers[i](
            #         query_embed,
            #         query_pos=query_pos,
            #     )

            # FFN
            query_embed = self.transformer_ffn_layers[i](query_embed)

        if img_ids is not None:
            num_imgs = len(img_ids)
            sp_labels = img_ids.unsqueeze(1).expand(num_imgs, num_imgs)
            img_labels = img_ids.unsqueeze(0).expand(num_imgs, num_imgs)
            pretrain_labels = (sp_labels == img_labels).to(torch.float32).view(-1, 1)
            return self.pretrain_task_head(query_embed.squeeze(0)), pretrain_labels

        # Prediction
        if self.transfer_learning_setting == 'none' or not self.project_queries:
            pred_queries = query_embed[:self.ntask]
        elif self.transfer_learning_setting == 'search2freeview':
            pred_queries = query_embed[:self.ntask].permute(1, 2, 0)
            pred_queries = self.search2freeview_mlp(pred_queries).permute(
                2, 0, 1)
            pred_queries += self.new_query_embed.weight.unsqueeze(1)
            num_fixs = num_fixs[:1]
        elif self.transfer_learning_setting == 'freeview2search':
            pred_queries = query_embed[:self.ntask].permute(1, 2, 0)
            pred_queries = self.freeview2search_mlp(pred_queries).permute(
                2, 0, 1)
            pred_queries += self.new_query_embed.weight.unsqueeze(1)
            num_fixs = num_fixs.expand(18, bs)
        else:
            raise NotImplementedError
        x = torch.cat([pred_queries, num_fixs.unsqueeze(-1)], dim=-1)
        output_termination = self.termination_predictor(x)
        out = {
            "pred_termination": output_termination.squeeze(-1).transpose(0, 1)
        }

        fixation_embed = self.fixation_embed(pred_queries)
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed,
                                            output_featmaps)
        if self.training:
            out["pred_fixation_map"] = outputs_fixation_map.transpose(0, 1)
        else:
            outputs_fixation_map = outputs_fixation_map[task_ids,
                                                        torch.arange(bs)]
            outputs_fixation_map = torch.sigmoid(outputs_fixation_map)
            outputs_fixation_map = F.interpolate(
                outputs_fixation_map.unsqueeze(1),
                size=(self.pa.im_h, self.pa.im_w)).squeeze(1)
            out["pred_fixation_map"] = outputs_fixation_map

        # Auxiliary tasks
        if self.training:
            ind = self.ntask
            if self.output_centermap:
                centermap_embed = self.centermap_embed(query_embed[ind:ind +
                                                                   80])
                out["pred_centermap"] = torch.einsum(
                    "lbc,bchw->lbhw", centermap_embed,
                    high_res_featmaps).transpose(0, 1)
                ind += 80
            if self.output_saliency:
                saliency_embed = self.saliency_embed(
                    # query_embed[ind: ind + 1])
                    query_embed[:self.ntask])
                outputs_saliency = torch.einsum("lbc,bchw->lbhw",
                                                saliency_embed,
                                                high_res_featmaps)
                out["pred_saliency"] = outputs_saliency.transpose(0, 1)
                ind += 1
            if self.output_target_map:
                target_embed = self.target_embed(query_embed[:self.ntask])
                out["pred_target_map"] = torch.einsum(
                    "lbc, bchw->lbhw", target_embed,
                    high_res_featmaps).transpose(0, 1)

        if return_attn_weights:
            out['cross_attn_weights'] = attn_weights_all_layers
        return out

    def encode(self, img: torch.Tensor):
        # Prepare dorsal embeddings
        img_embs_s4, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(img_embs_s4)
        if self.output_feature_map_name == 'P4':
            output_featmaps = high_res_featmaps
        elif self.output_feature_map_name == 'P2':
            output_featmaps = self.input_proj(img_embs_s2)
        else:
            raise NotImplementedError
        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        if "P1" in self.dorsal_source:
            # C x 10 x 16
            img_embs = self.input_proj(img_embs_s1)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=8)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[0].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        if "P2" in self.dorsal_source:
            # C x 20 x 32
            img_embs = self.input_pro1(img_embs_s2)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=4)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[1].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))

        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)
        dorsal_pos = (dorsal_pos, scale_embs)

        return dorsal_embs, dorsal_pos, None, (high_res_featmaps, output_featmaps)

        # high_res_featmaps, img_embs = self.encoder(img)
        # high_res_featmaps = self.input_proj(high_res_featmaps)
        # img_embs = self.input_proj(img_embs)
        # bs, c, h, w = high_res_featmaps.shape
        # img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)

        # # Add position encoding to feature maps
        # fm_pos = self.pixel_loc_emb(
        #     torch.arange(h * w).to(img.device).unsqueeze(1).expand(
        #         h * w, bs) + self.symbol_offset).permute(1, 2, 0).view(
        #             bs, c, h, w)
        # high_res_featmaps += fm_pos
        # return img_embs, high_res_featmaps

    def decode_and_predict(self,
                           dorsal_embs: torch.Tensor,
                           dorsal_pos: tuple,
                           dorsal_mask: torch.Tensor,
                           high_res_featmaps: tuple[torch.Tensor, torch.Tensor],
                           tgt_seq: torch.Tensor,
                           tgt_padding_mask: torch.Tensor,
                           tgt_seq_high: torch.Tensor,
                           task_ids: torch.Tensor = None,
                           return_attn_weights: bool = False):
        # Prepare ventral embeddings
        dorsal_pos, scale_embs = dorsal_pos
        high_res_featmaps, output_featmaps = high_res_featmaps
        bs, c = high_res_featmaps.shape[:2]
        highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
        # tgt_seq = tgt_seq.transpose(0,1)
        tgt_seq_high = tgt_seq_high.transpose(0, 1)
        if dorsal_mask is None:
            dorsal_mask = torch.zeros(1,
                                      *highres_embs.shape[1:],
                                      device=dorsal_embs.device)
        ventral_embs = torch.gather(
            torch.cat([dorsal_mask, highres_embs], dim=0), 0,
            tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape,
                                              highres_embs.size(-1)))
        ventral_pos = self.pixel_loc_emb(
            tgt_seq_high)  # Pos for fixation location

        # Add pos into embeddings for attention prediction
        if self.combine_pos_emb:
            # Dorsal embeddings
            dorsal_embs += dorsal_pos
            dorsal_pos = torch.zeros_like(dorsal_pos)
            # Ventral embeddings
            ventral_embs += ventral_pos
            ventral_pos = torch.zeros_like(ventral_pos)
            # High-res featmap
            # high_res_featmaps = self.visual_pos_fuse_layer(
            #     torch.cat([high_res_featmaps,
            #                self.pixel_loc_emb.forward_featmaps(
            #                    high_res_featmaps.shape[-2:]).expand_as(high_res_featmaps)], dim=-3))
            output_featmaps_wpos = output_featmaps + self.pixel_loc_emb.forward_featmaps(
                output_featmaps.shape[-2:], self.pos_scale)

        # Add embedding indicator embedding into pos embedding
        dorsal_pos += scale_embs
        ventral_pos += self.ventral_ind_emb.weight.unsqueeze(0).expand(
            *ventral_pos.shape)

        # Temporal embedding for fixations
        # ventral_pos += self.positional_encoding(
        #     ventral_embs).repeat(1, bs, 1)
        ventral_pos += self.fix_ind_emb.weight[:ventral_embs.
                                               size(0)].unsqueeze(1).repeat(
                                                   1, bs, 1)

        if self.combine_all_emb:
            dorsal_embs += dorsal_pos
            ventral_embs += ventral_pos
            dorsal_pos = ventral_pos = None

        # Update working memory with both dorsal and ventral memory
        # if using parallel architecture
        if self.parallel_arch:
            working_memory = torch.cat([dorsal_embs, ventral_embs], dim=0)
            padding_mask = torch.cat(
                [
                    torch.zeros(
                        bs, dorsal_embs.size(0),
                        device=dorsal_embs.device).bool(), tgt_padding_mask
                ],
                dim=1) if tgt_padding_mask is not None else None
            if self.combine_all_emb:
                working_memory_pos = None
            else:
                working_memory_pos = torch.cat([dorsal_pos, ventral_pos],
                                               dim=0)
            # working_memory = dorsal_embs # ventral_embs
            # padding_mask = None # tgt_padding_mask
            # working_memory_pos = dorsal_pos # ventral_pos

            if self.num_encoder_layers > 0:
                working_memory = self.working_memory_encoder(
                    working_memory,
                    src_key_padding_mask=padding_mask,
                    pos=working_memory_pos)
            dorsal_embs = working_memory
            dorsal_pos = working_memory_pos
        else:
            padding_mask = None

        # Update queries with attention
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pos = None
        num_fixs = torch.ones(self.ntask, bs).to(
            query_embed.device) * tgt_seq_high.size(0)
        # num_fix_embed = self.fix_ind_emb.weight[num_fixs.to(torch.long) - 1]
        # query_embed += num_fix_embed

        attn_weights_all_layers = []
        for i in range(self.num_decoder_layers):
            # Dorsal cross attention
            query_embed, attn_weights = self.transformer_cross_attention_layers_dorsal[
                i](query_embed,
                   dorsal_embs,
                   memory_mask=None,
                   memory_key_padding_mask=padding_mask,
                   pos=dorsal_pos,
                   query_pos=query_pos)
            if return_attn_weights:
                attn_weights_all_layers.append(attn_weights)

            if not self.parallel_arch:
                # Ventral cross attention
                query_embed, _ = self.transformer_cross_attention_layers_ventral[
                    i](query_embed,
                       ventral_embs,
                       memory_mask=None,
                       memory_key_padding_mask=tgt_padding_mask,
                       pos=ventral_pos,
                       query_pos=query_pos)
            # if self.ntask > 1:
            #     # Self attention
            #     query_embed = self.transformer_self_attention_layers[i](
            #         query_embed,
            #         query_pos=query_pos,
            #     )

            # FFN
            query_embed = self.transformer_ffn_layers[i](query_embed)

        # Prediction
        if self.transfer_learning_setting == 'none' or not self.project_queries:
            pred_queries = query_embed[:self.ntask]
        elif self.transfer_learning_setting == 'search2freeview':
            pred_queries = query_embed[:self.ntask].permute(1, 2, 0)
            pred_queries = self.search2freeview_mlp(pred_queries).permute(
                2, 0, 1)
            pred_queries += self.new_query_embed.weight.unsqueeze(1)
            num_fixs = num_fixs[:1]
        elif self.transfer_learning_setting == 'freeview2search':
            pred_queries = query_embed[:self.ntask].permute(1, 2, 0)
            pred_queries = self.freeview2search_mlp(pred_queries).permute(
                2, 0, 1)
            pred_queries += self.new_query_embed.weight.unsqueeze(1)
            num_fixs = num_fixs.expand(18, bs)
        else:
            raise NotImplementedError

        x = torch.cat([pred_queries, num_fixs.unsqueeze(-1)], dim=-1)
        output_termination = torch.sigmoid(self.termination_predictor(x))
        out = {
            "pred_termination":
            output_termination.squeeze(-1)[task_ids,
                                           torch.arange(bs)]
        }

        fixation_embed = self.fixation_embed(pred_queries)
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed,
                                            output_featmaps_wpos)
        outputs_fixation_map = torch.sigmoid(
            outputs_fixation_map[task_ids, torch.arange(bs)])
        outputs_fixation_map = F.interpolate(outputs_fixation_map.unsqueeze(1),
                                             size=(self.pa.im_h,
                                                   self.pa.im_w)).squeeze(1)
        out["pred_fixation_map"] = outputs_fixation_map.view(bs, -1)

        if return_attn_weights:
            out['cross_attn_weights'] = attn_weights_all_layers
        return out


# Foveated object memory
class FoveatedObjectMemory(nn.Module):
    def __init__(
        self,
        pa,
        num_decoder_layers: int,
        hidden_dim: int,
        nhead: int,
        ntask: int,
        num_output_layers: int,
        train_encoder: bool = False,
        pre_norm: bool = False,
        dropout: float = 0.1,
        dim_feedforward: int = 512,
        num_encoder_layers: int = 3,
    ):
        super(FoveatedObjectMemory, self).__init__()
        self.pa = pa
        self.num_decoder_layers = num_decoder_layers

        # Encoder: Deformable Attention Transformer
        self.train_encoder = train_encoder
        self.encoder = ImageFeatureEncoder(pa.backbone_config,
                                           dropout,
                                           pa.pixel_decoder,
                                           load_segm_decoder=True)
        self.symbol_offset = len(self.pa.special_symbols)
        if not train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        featmap_channels = 256
        if hidden_dim != featmap_channels:
            self.input_proj = nn.Conv2d(featmap_channels,
                                        hidden_dim,
                                        kernel_size=1)
            self.query_proj = nn.Linear(featmap_channels, hidden_dim)
            weight_init.c2_xavier_fill(self.input_proj)
            weight_init.c2_xavier_fill(self.query_proj)
        else:
            self.query_proj = self.input_proj = nn.Sequential()

        # Queries
        self.ntask = ntask
        self.query_embed = nn.Embedding(ntask, hidden_dim)
        self.query_pos = nn.Embedding(ntask, hidden_dim)

        # Fixation decoder
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))

        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=pre_norm)
            encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
            self.working_memory_encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Task heads
        self.termination_predictor = MLP(hidden_dim + 1, hidden_dim, 1,
                                         num_output_layers)
        self.fixation_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                  num_output_layers)
        self.pixel_loc_emb = pe.PositionalEncoding2D(pa,
                                                     hidden_dim,
                                                     height=pa.im_h // 4,
                                                     width=pa.im_w // 4,
                                                     dropout=dropout)
        self.query_level_embed = nn.Embedding(3, hidden_dim)
        self.num_area_bins = 100
        # self.mask_area_encoding = pe.PositionalEncoding(
        #     hidden_dim, self.num_area_bins)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2, stride=2),
            LayerNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(20, 32),
            nn.Conv2d(16, hidden_dim, kernel_size=1),
        )

    def enhance_queries(self, queries, segm_results):
        """Enhance queries with segmentation map size and positions"""
        obj_query_mask = torch.stack([x[2] for x in segm_results], dim=1)
        # Encoding mask using CNN
        masks = []
        for seg_rst in segm_results:
            panoptic_mask = seg_rst[0]
            for seg in seg_rst[1]:
                masks.append(panoptic_mask == seg['id'])
        masks = torch.stack(masks, dim=0).to(queries.device,
                                             dtype=torch.float32).unsqueeze(1)
        mask_embs = self.mask_encoder(masks).squeeze()
        queries[:, obj_query_mask] += mask_embs
        # Encoding mask area, position and size
        # mask_area, mask_centroid = [], []
        # for seg_rst in segm_results:
        #     for seg in seg_rst[1]:
        # mask_area.append(seg['mask_area'])
        # mask_centroid.append(seg['mask_centroid'])
        # mask_area = torch.tensor(mask_area, dtype=torch.float)
        # mask_centroid = torch.tensor(mask_centroid, dtype=torch.float)
        # mask_area_ids = ((mask_area - .1) /
        #                  (self.pa.im_h // 4 * self.pa.im_w // 4) *
        #                  self.num_area_bins).long()
        # mask_area_embed = self.mask_area_encoding.forward_pos(mask_area_ids)
        # mask_pos_embd = self.pixel_loc_emb.forward_pos(mask_centroid[:, 0],
        #                                                mask_centroid[:, 1])
        # queries[:, obj_query_mask] += mask_area_embed + mask_pos_embd

        return queries, torch.logical_not(
            obj_query_mask.unsqueeze(0).expand(queries.shape[:3]))

    def encode(self, img: torch.Tensor):
        high_res_featmaps, segm_queries, segm_results = self.encoder(img)
        high_res_featmaps = self.input_proj(high_res_featmaps)
        working_memory, working_memory_pos, working_memory_mask = self.construct_foveated_object_memory(
            segm_queries, segm_results)
        return working_memory, working_memory_pos, working_memory_mask, high_res_featmaps

    def construct_foveated_object_memory(self, queries, segm_results):
        """Construct foveated object memory from object queries."""
        fom = torch.stack(queries, dim=0)
        fom = self.query_proj(fom)

        # Enhance queries with segmentation map size and positions
        fom, fom_mask = self.enhance_queries(fom, segm_results)

        # Add query level embedding
        for i in range(len(fom)):
            fom[i] += self.query_level_embed.weight[i]

        fom_pos = None
        return fom.view(-1, *fom.shape[2:]), fom_pos, fom_mask.reshape(
            -1, fom_mask.shape[-1]).transpose(0, 1)

    def decode_and_predict(self,
                           working_memory: torch.Tensor,
                           working_memory_pos: torch.Tensor,
                           working_memory_mask: torch.Tensor,
                           high_res_featmaps: torch.Tensor,
                           tgt_seq: torch.Tensor,
                           tgt_padding_mask: torch.Tensor,
                           tgt_seq_high: torch.Tensor,
                           task_ids: torch.Tensor = None,
                           return_attn_weights: bool = False):
        """Predict fixation probability map from foveated object memory."""
        out = {}

        bs = working_memory.size(1)
        if tgt_seq_high.size(0) == bs:
            tgt_seq_high = tgt_seq_high.transpose(0, 1)

        # Augment working memory with fixation history
        fix_pos_emb = self.pixel_loc_emb(tgt_seq_high)
        if working_memory_mask is None:
            working_memory_mask = torch.zeros(
                bs, working_memory.size(0),
                device=working_memory.device).bool()
        padding_mask = torch.cat(
            [working_memory_mask, tgt_padding_mask],
            dim=1) if tgt_padding_mask is not None else None
        working_memory = torch.cat([working_memory, fix_pos_emb], dim=0)

        # Learning foveated object memory
        if self.num_encoder_layers > 0:
            working_memory = self.working_memory_encoder(
                working_memory,
                src_key_padding_mask=padding_mask,
                pos=working_memory_pos)

        # Update queries with attention
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1)

        # Let model know which fixation it's going to predict
        if tgt_padding_mask is not None:
            num_fixs = (tgt_padding_mask.size(1) -
                        torch.sum(tgt_padding_mask, dim=1)).unsqueeze(0).expand(
                            self.ntask, bs)
        else:
            num_fixs = torch.ones(1, bs).to(
                query_embed.device) * tgt_seq_high.size(0)

        attn_weights_all_layers = []
        for i in range(self.num_decoder_layers):
            query_embed, attn_weights = self.transformer_cross_attention_layers[
                i](query_embed,
                   working_memory,
                   memory_mask=None,
                   memory_key_padding_mask=padding_mask,
                   pos=working_memory_pos,
                   query_pos=query_pos)
            if return_attn_weights:
                attn_weights_all_layers.append(attn_weights)

            if self.ntask > 1:
                # Self attention
                query_embed = self.transformer_self_attention_layers[i](
                    query_embed,
                    query_pos=query_pos,
                )

            # FFN
            query_embed = self.transformer_ffn_layers[i](query_embed)

        # Prediction
        x = torch.cat([query_embed[:self.ntask],
                       num_fixs.unsqueeze(-1)],
                      dim=-1)
        output_termination = self.termination_predictor(x)
        fixation_embed = self.fixation_embed(query_embed[:self.ntask])
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed,
                                            high_res_featmaps)

        if self.training:
            out = {
                "pred_termination":
                output_termination.squeeze(-1).transpose(0, 1)
            }
            out["pred_fixation_map"] = outputs_fixation_map.transpose(0, 1)
        else:
            assert task_ids is not None, "task_ids must be provided during inference."
            out = {
                "pred_termination":
                output_termination.squeeze(-1)[task_ids,
                                               torch.arange(bs)]
            }
            outputs_fixation_map = outputs_fixation_map[task_ids,
                                                        torch.arange(bs)]
            outputs_fixation_map = torch.sigmoid(outputs_fixation_map)
            outputs_fixation_map = F.interpolate(
                outputs_fixation_map.unsqueeze(1),
                size=(self.pa.im_h, self.pa.im_w)).squeeze(1)
            out["pred_fixation_map"] = outputs_fixation_map
        return out

    def forward(self,
                img: torch.Tensor,
                tgt_seq: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                tgt_seq_high: torch.Tensor,
                task_ids: torch.Tensor = None,
                return_attn_weights: bool = False):
        tgt_seq_high = tgt_seq_high.transpose(0, 1)
        fom, fom_pos, fom_mask, high_res_featmaps = self.encode(img)
        out = self.decode_and_predict(fom, fom_pos, fom_mask,
                                      high_res_featmaps, tgt_seq,
                                      tgt_padding_mask, tgt_seq_high, task_ids,
                                      return_attn_weights)
        return out


# Dense prediction transformer
class HumanAttnTransformerV2(nn.Module):
    def __init__(
        self,
        pa,
        num_decoder_layers: int,
        hidden_dim: int,
        nhead: int,
        ntask: int,
        num_output_layers: int,
        train_encoder: bool = False,
        pre_norm: bool = False,
        dropout: float = 0.1,
        dim_feedforward: int = 512,
        parallel_arch: bool = False,
        dorsal_source: list = ["P2"],
        num_encoder_layers: int = 3,
        output_centermap: bool = False,
        output_saliency: bool = False,
        output_target_map: bool = False,
        combine_pos_emb: bool = True,
        combine_all_emb: bool = False,
        share_task_emb: bool = True,
    ):
        super(HumanAttnTransformerV2, self).__init__()
        self.pa = pa
        self.num_decoder_layers = num_decoder_layers
        self.combine_pos_emb = combine_pos_emb
        self.combine_all_emb = combine_all_emb
        self.parallel_arch = parallel_arch
        self.share_task_emb = share_task_emb
        self.dorsal_source = dorsal_source
        assert len(dorsal_source) > 0, "need to specify dorsal source: P1, P2!"
        self.output_centermap = output_centermap
        self.output_saliency = output_saliency
        self.output_target_map = output_target_map

        # Encoder: Deformable Attention Transformer
        self.train_encoder = train_encoder
        self.encoder = ImageFeatureEncoder(pa.backbone_config, dropout,
                                           pa.pixel_decoder)
        self.symbol_offset = len(self.pa.special_symbols)
        if not train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        featmap_channels = 256
        if hidden_dim != featmap_channels:
            self.input_proj = nn.Conv2d(featmap_channels,
                                        hidden_dim,
                                        kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()

        # Queries
        self.ntask = ntask
        self.aux_queries = 0
        if self.output_centermap:
            self.aux_queries += 80
        if self.output_saliency:
            self.aux_queries += 1
        self.query_embed = nn.Embedding(ntask + self.aux_queries, hidden_dim)

        # Decoder
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_dorsal = nn.ModuleList()
        self.transformer_cross_attention_layers_ventral = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            self.transformer_cross_attention_layers_dorsal.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            if not self.parallel_arch:
                self.transformer_cross_attention_layers_ventral.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    ))
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))

        self.num_encoder_layers = num_encoder_layers
        if self.parallel_arch and num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=pre_norm)
            encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
            self.working_memory_encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Task heads
        self.termination_predictor = MLP(hidden_dim + 1, hidden_dim, 1,
                                         num_output_layers)
        self.fixation_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                  num_output_layers)
        if self.output_target_map:
            self.target_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                    num_output_layers)
        if self.output_centermap:
            self.centermap_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                       num_output_layers)
        if self.output_saliency:
            self.saliency_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                      num_output_layers)

        # Positional embedding
        self.pixel_loc_emb = pe.PositionalEncoding2D(pa,
                                                     hidden_dim,
                                                     height=pa.im_h // 4,
                                                     width=pa.im_w // 4,
                                                     dropout=dropout)

        self.fix_ind_emb = nn.Embedding(pa.max_traj_length, hidden_dim)
        if self.share_task_emb:
            self.task_emb = self.query_embed
        else:
            self.task_emb = nn.Embedding(ntask, hidden_dim)

        # Embedding for distinguishing dorsal or ventral embeddings
        self.dorsal_ind_emb = nn.Embedding(2, hidden_dim)  # P1 and P2
        self.ventral_ind_emb = nn.Embedding(1, hidden_dim)

    def encode(self, img: torch.Tensor):
        # Prepare dorsal embeddings
        high_res_featmaps, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(high_res_featmaps)
        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        if "P1" in self.dorsal_source:
            # C x 10 x 16
            img_embs = self.input_proj(img_embs_s1)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=8)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[0].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        if "P2" in self.dorsal_source:
            # C x 20 x 32
            img_embs = self.input_proj(img_embs_s2)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=4)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[1].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)
        dorsal_pos = (dorsal_pos, scale_embs)
        return dorsal_embs, dorsal_pos, None, high_res_featmaps

    def construct_working_memory(self, dorsal_embs: torch.Tensor,
                                 dorsal_pos: tuple,
                                 tgt_padding_mask: torch.Tensor,
                                 high_res_featmaps: torch.Tensor,
                                 tgt_seq_high: torch.Tensor,
                                 task_ids: torch.Tensor):
        dorsal_pos, scale_embs = dorsal_pos
        bs, c = high_res_featmaps.shape[:2]
        # Prepare ventral embeddings
        highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
        ventral_embs = torch.gather(
            torch.cat([
                torch.zeros(1,
                            *highres_embs.shape[1:],
                            device=dorsal_embs.device), highres_embs
            ],
                      dim=0), 0,
            tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape,
                                              highres_embs.size(-1)))
        ventral_pos = self.pixel_loc_emb(
            tgt_seq_high)  # Pos for fixation location

        # Add pos into embeddings for attention prediction
        if self.combine_pos_emb:
            # Dorsal embeddings
            dorsal_embs += dorsal_pos
            dorsal_pos.fill_(0)
            # Ventral embeddings
            ventral_embs += ventral_pos
            ventral_pos.fill_(0)
            # High-res featmap
            high_res_featmaps += self.pixel_loc_emb.forward_featmaps(
                high_res_featmaps.shape[-2:])

        # Add embedding indicator embedding into pos embedding
        dorsal_pos += scale_embs
        ventral_pos += self.ventral_ind_emb.weight.unsqueeze(0).expand(
            *ventral_pos.shape)

        # Temporal embedding for fixations
        ventral_pos += self.fix_ind_emb.weight[:ventral_embs.
                                               size(0)].unsqueeze(1).repeat(
                                                   1, bs, 1)

        # Task embedding
        task_embs = self.task_emb.weight[task_ids].unsqueeze(0)
        dorsal_pos += task_embs
        ventral_pos += task_embs

        if tgt_padding_mask is not None:
            ventral_pos[tgt_padding_mask.transpose(0, 1)] = 0

        if self.combine_all_emb:
            dorsal_embs += dorsal_pos
            ventral_embs += ventral_pos
            working_memory_pos = None
        else:
            working_memory_pos = torch.cat([dorsal_pos, ventral_pos], dim=0)

        # Update working memory
        working_memory = torch.cat([dorsal_embs, ventral_embs], dim=0)
        padding_mask = torch.cat(
            [
                torch.zeros(bs, dorsal_embs.size(0),
                            device=dorsal_embs.device).bool(), tgt_padding_mask
            ],
            dim=1) if tgt_padding_mask is not None else None

        working_memory = self.working_memory_encoder(
            working_memory,
            src_key_padding_mask=padding_mask,
            pos=working_memory_pos)

        return working_memory, working_memory_pos, padding_mask

    def decode_and_predict(self,
                           dorsal_embs: torch.Tensor,
                           dorsal_pos: tuple,
                           dorsal_mask: torch.Tensor,
                           high_res_featmaps: torch.Tensor,
                           tgt_seq: torch.Tensor,
                           tgt_padding_mask: torch.Tensor,
                           tgt_seq_high: torch.Tensor,
                           task_ids: torch.Tensor,
                           return_attn_weights: bool = False):
        bs = high_res_featmaps.size(0)
        if tgt_seq_high.size(0) == bs:
            tgt_seq_high = tgt_seq_high.transpose(0, 1)

        working_memory, working_memory_pos, working_memory_mask = self.construct_working_memory(
            dorsal_embs, dorsal_pos, tgt_padding_mask, high_res_featmaps,
            tgt_seq_high, task_ids)

        # Update queries with attention
        query_embed = self.query_embed.weight[task_ids].unsqueeze(0)

        # Encode previous fixations spatially and temporally
        fixation_pos_emb = self.pixel_loc_emb(tgt_seq_high)
        fixation_ind_emb = self.fix_ind_emb.weight[:tgt_seq_high.size(
            0)].unsqueeze(1).repeat(1, bs, 1)
        fixation_history_emb = fixation_pos_emb + fixation_ind_emb

        # Use fixation history as queries (following the decoder of SAM)
        query_embed = torch.cat([query_embed, fixation_history_emb], dim=0)
        query_mask = torch.cat(
            [
                torch.zeros(bs, 1,
                            device=dorsal_embs.device).bool(), tgt_padding_mask
            ],
            dim=1) if tgt_padding_mask is not None else None

        attn_weights_all_layers = []
        for i in range(self.num_decoder_layers):
            # Dorsal cross attention
            query_embed, attn_weights = self.transformer_cross_attention_layers_dorsal[
                i](query_embed,
                   working_memory,
                   memory_mask=None,
                   memory_key_padding_mask=working_memory_mask,
                   pos=working_memory_pos,
                   query_pos=None)
            if return_attn_weights:
                attn_weights_all_layers.append(attn_weights)

            # Self attention
            query_embed = self.transformer_self_attention_layers[i](
                query_embed,
                tgt_key_padding_mask=query_mask,
                query_pos=None,
            )

            # FFN
            query_embed = self.transformer_ffn_layers[i](query_embed)

        # Prediction
        # Let model know which fixation it's going to predict
        if tgt_padding_mask is not None:
            num_fixs = (tgt_padding_mask.size(1) -
                        torch.sum(tgt_padding_mask, dim=1)).unsqueeze(0)
        else:
            num_fixs = torch.ones(1, bs).to(
                query_embed.device) * tgt_seq_high.size(0)
        pred_queries = query_embed[:1]

        x = torch.cat([pred_queries, num_fixs.unsqueeze(-1)], dim=-1)
        output_termination = torch.sigmoid(self.termination_predictor(x))
        out = {"pred_termination": output_termination.squeeze(-1).squeeze(0)}

        fixation_embed = self.fixation_embed(pred_queries)
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed,
                                            high_res_featmaps).squeeze(0)
        if not self.training:
            outputs_fixation_map = torch.sigmoid(outputs_fixation_map)
            outputs_fixation_map = F.interpolate(
                outputs_fixation_map.unsqueeze(1),
                size=(self.pa.im_h, self.pa.im_w)).squeeze(1)
        out["pred_fixation_map"] = outputs_fixation_map

        if return_attn_weights:
            out['cross_attn_weights'] = attn_weights_all_layers
        return out

    def forward(self,
                img: torch.Tensor,
                tgt_seq: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                tgt_seq_high: torch.Tensor,
                task_ids: torch.Tensor,
                return_attn_weights=False):
        dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps = self.encode(
            img)
        out = self.decode_and_predict(dorsal_embs, dorsal_pos, dorsal_mask,
                                      high_res_featmaps, tgt_seq,
                                      tgt_padding_mask, tgt_seq_high, task_ids,
                                      return_attn_weights)
        return out
