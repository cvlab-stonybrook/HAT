"""
Two-pathway (Ventral and Dorsal) Transformer Training Script.
This script is a simplified version of the training scripts in 
https://github.com/cvlab-stonybrook/Scanpath_Prediction
"""
import argparse
import os
import random

import numpy as np

import datetime
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from collections import defaultdict
from hat.builder import build
from common.config import JsonConfig
from common.losses import focal_loss
from common.utils import (
    transform_fixations, )
from hat.evaluation import evaluate
# from common.sinkhorn import SinkhornDistance

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams',
                        type=str,
                        help='hyper parameters config file path')
    parser.add_argument('--dataset-root', type=str, help='dataset root path')
    parser.add_argument('--pretrain',
                        action='store_true',
                        help='perform pretraining')
    parser.add_argument('--model',
                        choices=['HAT', 'FOM', 'HATv2'],
                        default='HAT',
                        help='model type')
    parser.add_argument('--transfer-learn',
                        choices=['none', 'search2freeview', 'freeview2search', 'finetune'],
                        default='none',
                        help='setting for transfer learning')
    parser.add_argument('--eval-only',
                        action='store_true',
                        help='perform evaluation only')
    parser.add_argument(
        '--split',
        type=int,
        default=1,
        help='dataset split for MIT1003/CAT2000 only (default=1)')
    parser.add_argument(
        '--eval-mode',
        choices=['greedy', 'sample'],
        type=str,
        default='greedy',
        help=
        'whether to sample scanapth or greedily predict scanpath during evaluation (default=greedy)'
    )
    parser.add_argument('--disable-saliency',
                        action='store_true',
                        help='do not calculate saliency metrics')

    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='gpu id (default=0)')
    return parser.parse_args()


def log_dict(writer, scalars, step, prefix):
    for k, v in scalars.items():
        writer.add_scalar(prefix + "/" + k, v, step)


def compute_loss(model, batch, losses, loss_funcs, pa):
    fix_losses, aux_losses = 0, 0
    img = batch['true_state'].to(device)
    if args.pretrain and len(losses) == 1 and losses[0] == 'saliency_pred':
        # Only saliency prediction loss
        logits = model(img)
    else:
        task_ids = batch['task_id'].to(device)
        is_last = batch['is_last'].to(device)
        IOR_weight_map = batch['IOR_weight_map'].to(device)
        is_fv = batch['is_freeview'].to(device)
        not_fv = torch.logical_not(is_fv)
        inp_seq, inp_seq_high = transform_fixations(batch['normalized_fixations'],
                                                    batch['is_padding'],
                                                    hparams.Data,
                                                    False,
                                                    return_highres=True)
        inp_seq = inp_seq.to(device)
        inp_padding_mask = (inp_seq == pa.pad_idx)
        if args.pretrain and 'mismatch_cls' in losses:
            temp = defaultdict(lambda: len(temp))
            img_ids = torch.tensor([temp[ele] for ele in batch['img_name']], dtype=torch.int16).to(device)
        else:
            img_ids = None
        logits = model(img, inp_seq, inp_padding_mask, inp_seq_high.to(device),
                    task_ids, img_ids=img_ids)

    bs = img.size(0)
    loss_dict = {}
    if args.pretrain and "mismatch_cls" in losses:
        logits, labels = logits
        loss_dict['mismatch_cls'] = loss_funcs['mismatch_cls'](logits, labels)

    if not args.pretrain:
        is_all_task = len(logits['pred_fixation_map'].size()) > 3
    if "next_fix_pred" in losses:
        # Next fixation prediction
        non_term_mask = torch.logical_not(is_last)
        if not args.pretrain:
            if is_all_task:
                pred_fix_map = logits['pred_fixation_map'][torch.arange(bs),
                                                            task_ids]
            else:
                pred_fix_map = logits['pred_fixation_map']
        else:
            pred_fix_map = logits['pred_fixation_map'].squeeze(1)
        if use_focal_loss:
            pred_fix_map = torch.sigmoid(pred_fix_map)
        if pred_fix_map.size(-1) != pa.im_w:
            pred_fix_map = F.interpolate(pred_fix_map.unsqueeze(1),
                                            size=(pa.im_h,
                                                pa.im_w)).squeeze(1)
        tgt_fix_map = batch['target_fix_map'].to(device)
        pred_fix_map = pred_fix_map[non_term_mask]
        tgt_fix_map = tgt_fix_map[non_term_mask]

        loss_dict['next_fix_pred'] = loss_funcs['next_fix_pred'](
            pred_fix_map,
            tgt_fix_map,
            alpha=1,
            beta=4,
            weights=IOR_weight_map[non_term_mask])

        # not_fv = not_fv[non_term_mask]
        # is_fv = is_fv[non_term_mask]
        # loss_dict['next_fix_pred'] = loss_funcs['next_fix_pred'](
        #     pred_fix_map[not_fv], tgt_fix_map[not_fv])
        # if torch.any(is_fv) and hparams.Data.TAP !='FV':
        #     loss_dict['next_fix_pred'] += loss_funcs['next_fix_pred'](
        #         pred_fix_map[is_fv], tgt_fix_map[is_fv]) / 36

    if "term_pred" in losses:
        # Termination prediction
        if is_all_task:
            pred_termination = logits['pred_termination'][
                torch.arange(bs), task_ids]
        else:
            pred_termination = logits['pred_termination']
        loss_dict['term_pred'] = loss_funcs['term_pred'](
            pred_termination, is_last.float())

    if "centermap_pred" in losses:
        pred_cm_map = torch.sigmoid(logits['pred_centermap'])
        tgt_cm_map = batch['centermaps'].to(device)
        pred_cm_map = F.interpolate(pred_cm_map, size=(pa.im_h, pa.im_w))
        loss_dict['centermap_pred'] = loss_funcs['centermap_pred'](
            pred_cm_map, tgt_cm_map)

    if "target_map_pred" in losses:
        pred_target_map = torch.sigmoid(
            logits['pred_target_map'][torch.arange(bs), task_ids])
        tgt_target_map = batch['label_coding'].to(device)
        pred_target_map = F.interpolate(pred_target_map.unsqueeze(1),
                                        size=(pa.im_h, pa.im_w)).squeeze(1)
        loss_dict['target_map_pred'] = loss_funcs['target_map_pred'](
            pred_target_map, tgt_target_map)

    if "saliency_pred" in losses:
        pred_sal_map = logits['pred_saliency']
        tgt_sal_map = batch['saliency_map'].to(device)
        tgt_sal_map = F.interpolate(tgt_sal_map,
                                     size=pred_sal_map.shape[-2:])
        if pred_sal_map.size(1) > 1:
            tgt_sal_map = tgt_sal_map.repeat(1, pred_sal_map.size(1), 1, 1)
        loss_dict['saliency_pred'] = loss_funcs['saliency_pred'](
            pred_sal_map.squeeze(), tgt_sal_map.squeeze())

    # if "task_pred" in losses:
    #     # Task prediction
    #     terminal_ind = tgt_input==pa.eos_idx
    #     task_loss = loss_funcs['task_pred'](
    #         logits[2][terminal_ind], task_ids[terminal_ind.sum(dim=0) > 0])
    #     loss_dict['task_pred'] = task_loss

    # if "mismatch_cls" in losses:
    #     # Scanpath-image matchness prediction
    #     mismatch_img = batch['mismatch_img'].to(device)
    #     mismatch_logits = model(img, tgt_input, task_ids, tgt_mask, tgt_padding_mask)
    #     mismatch_pred = torch.cat([
    #         logits[2][terminal_ind], mismatch_logits[2][terminal_ind]], dim=0)
    #     mismatch_target = torch.ones(mismatch_pred.size(0)).to(device)
    #     num_pos = terminal_ind.sum()
    #     mismatch_target[num_pos:] = 0
    #     mismatch_loss = loss_funcs['mismatch_cls'](
    #         mismatch_pred.squeeze(), mismatch_target)
    #     loss_dict['mismatch_cls'] = mismatch_loss
    return loss_dict


def train_iter(model, optimizer, batch, losses, loss_weights, loss_funcs, pa):
    assert len(losses) > 0, "no loss func assigned!"
    model.train()
    optimizer.zero_grad()

    loss_dict = compute_loss(model, batch, losses, loss_funcs, pa)
    loss = 0
    for k, v in loss_dict.items():
        loss += v * loss_weights[k]
    loss.backward()
    optimizer.step()

    for k in loss_dict:
        loss_dict[k] = loss_dict[k].item()

    return loss_dict


def get_eval_loss(model, eval_dataloader, losses, loss_funcs, pa):
    with torch.no_grad():
        model.eval()
        num_batches = 0
        avg_loss_dict = defaultdict(lambda: 0)
        for batch in tqdm(eval_dataloader, desc="computing eval loss"):
            loss_dict = compute_loss(model, batch, losses, loss_funcs, pa)
            for k in loss_dict:
                avg_loss_dict[k] += loss_dict[k].item()
            num_batches += 1
            if num_batches > 1000:
                break
        for k in avg_loss_dict:
            avg_loss_dict[k] /= num_batches
        return avg_loss_dict


def run_evaluation():
    # Perform evaluation
    rst_tp, rst_ta, rst_fv = None, None, None
    pred_tp = pred_ta = pred_fv = None
    if hparams.Data.TAP in ['TP', 'TAP', 'ALL']:
        rst_tp, pred_tp = evaluate(
            model,
            device,
            valid_img_loader_tp,
            valid_gaze_loader_tp,
            hparams_tp.Data,
            bbox_annos,
            human_cdf,
            fix_clusters,
            prior_maps_tp,
            sss_strings,
            dataset_root,
            sps_test_tp,
            sample_action=sample_action,
            output_saliency_metrics=output_saliency_metrics,
            center_initial=hparams.Data.name
            in ['COCO-Search18', 'COCO-Freeview'],
            log_dir=log_dir)
        print("TP:", rst_tp)
    if hparams.Data.TAP in ['TA', 'TAP', 'ALL']:
        rst_ta, pred_ta = evaluate(
            model,
            device,
            valid_img_loader_ta,
            valid_gaze_loader_ta,
            hparams_ta.Data,
            bbox_annos,
            human_cdf,
            fix_clusters,
            prior_maps_ta,
            sss_strings,
            dataset_root,
            sps_test_ta,
            sample_action=sample_action,
            output_saliency_metrics=output_saliency_metrics,
            center_initial=hparams.Data.name
            in ['COCO-Search18', 'COCO-Freeview'],
            log_dir=log_dir)
        print("TA", rst_ta)
    if hparams.Data.TAP in ['FV', 'ALL']:
        rst_fv, pred_fv = evaluate(
            model,
            device,
            valid_img_loader_fv,
            valid_gaze_loader_fv,
            hparams_fv.Data,
            bbox_annos,
            human_cdf,
            fix_clusters,
            prior_maps_fv,
            sss_strings,
            dataset_root,
            sps_test_fv,
            sample_action=sample_action,
            output_saliency_metrics=output_saliency_metrics,
            center_initial=hparams.Data.name
            in ['COCO-Search18', 'COCO-Freeview'],
            log_dir=log_dir)
        print("FV", rst_fv)
    return rst_tp, rst_ta, rst_fv


if __name__ == '__main__':
    args = parse_args()
    hparams = JsonConfig(args.hparams)
    hparams.Model.name = args.model
    hparams.Train.transfer_learn = args.transfer_learn
    dir = os.path.dirname(args.hparams)
    hparams_tp = JsonConfig(os.path.join(dir, 'coco_search18_dense_SSL_TP.json'))
    hparams_ta = JsonConfig(os.path.join(dir, 'coco_search18_dense_SSL_TA.json'))
    hparams_fv = JsonConfig(os.path.join(dir, 'coco_freeview_dense_SSL.json'))

    dataset_root = args.dataset_root
    if dataset_root[-1] == '/':
        dataset_root = dataset_root[:-1]
    output_saliency_metrics = not args.disable_saliency
    device = torch.device(f'cuda:{args.gpu_id}')
    sample_action = args.eval_mode == 'sample'
    if hparams.Data.name in ['MIT1003', 'CAT2000']:
        hparams.Train.log_dir += f'_split{args.split}'

    model, optimizer, train_gaze_loader, val_gaze_loader, train_img_loader, \
        valid_img_loader_tp, valid_img_loader_ta, valid_img_loader_fv, \
        global_step, bbox_annos, human_cdf, fix_clusters, prior_maps_tp, \
        prior_maps_ta, prior_maps_fv, sss_strings, valid_gaze_loader_tp, \
        valid_gaze_loader_ta, valid_gaze_loader_fv, sps_test_tp, \
        sps_test_ta, sps_test_fv, term_pos_weight, _ = build(
            hparams, dataset_root, device, args.pretrain, args.eval_only, args.split)

    log_dir = hparams.Train.log_dir
    if args.eval_only:
        run_evaluation()
    else:
        writer = SummaryWriter(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print("Log dir:", log_dir)
        log_folder_runs = "./runs/{}".format(log_dir.split('/')[-1])
        if not os.path.exists(log_folder_runs):
            os.system(f"mkdir -p {log_folder_runs}")

        # Write configuration file to the log dir
        hparams.dump(log_dir, 'config.json')

        print_every = 20
        max_iters = hparams.Train.max_iters
        save_every = hparams.Train.checkpoint_every
        eval_every = hparams.Train.evaluate_every
        pad_idx = hparams.Data.pad_idx
        use_focal_loss = hparams.Train.use_focal_loss
        loss_funcs = {
            "next_fix_pred":
            focal_loss if use_focal_loss else
            torch.nn.BCEWithLogitsLoss(),  # torch.nn.MSELoss(),
            "centermap_pred":
            focal_loss,
            "target_map_pred":
            focal_loss,
            "saliency_pred":
            torch.nn.BCEWithLogitsLoss(),
            "task_pred":
            torch.nn.CrossEntropyLoss(),
            "mismatch_cls":
            torch.nn.BCEWithLogitsLoss(),
            "term_pred":
            torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(term_pos_weight, dtype=torch.float32)),
        }

        loss_weights = {
            "next_fix_pred": 1.0,
            "centermap_pred": hparams.Train.centermap_pred_weight,
            "target_map_pred": hparams.Train.centermap_pred_weight,
            "saliency_pred": hparams.Train.saliency_pred_weight,
            "task_pred": hparams.Train.task_pred_weight,
            "mismatch_cls": hparams.Train.mismatch_cls,
            "term_pred": hparams.Train.term_pred_weight,
        }
        losses = hparams.Train.losses
        loss_dict_avg = dict(zip(losses, [0] * len(losses)))
        print("loss weights:", loss_weights)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=hparams.Train.lr_steps, gamma=0.1)

        s_epoch = int(global_step / len(train_gaze_loader))

        # loss_dict_avg = get_eval_loss(model, val_gaze_loader, losses,
        #                               loss_funcs, hparams_tp.Data)
        # log_dict(writer, loss_dict_avg, global_step, 'eval')
        last_time = datetime.datetime.now()
        for i_epoch in range(s_epoch, int(1e5)):
            scheduler.step()
            for i_batch, batch in enumerate(train_gaze_loader):
                loss_dict = train_iter(model, optimizer, batch, losses,
                                       loss_weights, loss_funcs, hparams.Data)
                for k in loss_dict:
                    loss_dict_avg[k] += loss_dict[k]

                if global_step % print_every == print_every - 1:
                    for k in loss_dict_avg:
                        loss_dict_avg[k] /= print_every

                    time = datetime.datetime.now()
                    eta = str((time - last_time) / print_every *
                              (max_iters - global_step))
                    last_time = time
                    time = str(time)
                    log_msg = "[{}], eta: {}, iter: {}, progress: {:.2f}%, epoch: {}, total loss: {:.3f}".format(
                        time[time.rfind(' ') + 1:time.rfind('.')],
                        eta[:eta.rfind('.')],
                        global_step,
                        (global_step / max_iters) * 100,
                        i_epoch,
                        np.sum(list(loss_dict_avg.values())),
                    )

                    for k, v in loss_dict_avg.items():
                        log_msg += " {}_loss: {:.3f}".format(k, v)

                    print(log_msg)
                    log_dict(writer, loss_dict_avg, global_step, 'train')
                    writer.add_scalar('train/lr',
                                      optimizer.param_groups[0]["lr"],
                                      global_step)
                    for k in loss_dict_avg:
                        loss_dict_avg[k] = 0

                # Evaluate
                if global_step % eval_every == eval_every - 1:
                    if args.pretrain:
                        loss_dict_avg = get_eval_loss(model, val_gaze_loader,
                                                      losses, loss_funcs,
                                                      hparams_tp.Data)
                        log_dict(writer, loss_dict_avg, global_step, 'eval')
                    else:
                        rst_tp, rst_ta, rst_fv = run_evaluation()
                        if rst_tp is not None:
                            log_dict(writer, rst_tp, global_step, "eval_TP")
                        if rst_ta is not None:
                            log_dict(writer, rst_ta, global_step, "eval_TA")
                        if rst_fv is not None:
                            log_dict(writer, rst_fv, global_step, "eval_FV")

                    writer.add_scalar('train/epoch',
                                      global_step / len(train_gaze_loader),
                                      global_step)
                    os.system(f"cp {log_dir}/events* {log_folder_runs}")

                if global_step % save_every == save_every - 1:
                    save_path = os.path.join(log_dir, f"ckp_{global_step}.pt")
                    if isinstance(model, torch.nn.DataParallel):
                        model_weights = model.module.state_dict()
                    else:
                        model_weights = model.state_dict()
                    torch.save(
                        {
                            'model': model_weights,
                            'optimizer': optimizer.state_dict(),
                            'step': global_step + 1,
                        },
                        save_path,
                    )
                    print(f"Saved checkpoint to {save_path}.")
                global_step += 1
                if global_step >= max_iters:
                    print("Exit program!")
                    break
            else:
                continue
            break  # Break outer loop

        # Copy to log file to ./runs
        os.system(f"cp {log_dir}/events* {log_folder_runs}")
