import scipy.ndimage as filters
import numpy as np
from .multimatch import docomparison
from . import utils
import torch
import gzip
from os.path import join


def multimatch(s1, s2, im_size):
    s1x = s1['X']
    s1y = s1['Y']
    l1 = len(s1x)
    if l1 < 3:
        scanpath1 = np.ones((3, 3), dtype=np.float32)
        scanpath1[:l1, 0] = s1x
        scanpath1[:l1, 1] = s1y
    else:
        scanpath1 = np.ones((l1, 3), dtype=np.float32)
        scanpath1[:, 0] = s1x
        scanpath1[:, 1] = s1y
    s2x = s2['X']
    s2y = s2['Y']
    l2 = len(s2x)
    if l2 < 3:
        scanpath2 = np.ones((3, 3), dtype=np.float32)
        scanpath2[:l2, 0] = s2x
        scanpath2[:l2, 1] = s2y
    else:
        scanpath2 = np.ones((l2, 3), dtype=np.float32)
        scanpath2[:, 0] = s2x
        scanpath2[:, 1] = s2y
    mm = docomparison(scanpath1, scanpath2, sz=im_size)
    return mm[0]


def compute_mm(human_trajs, model_trajs, im_w, im_h, tasks=None):
    """
    compute scanpath similarity using multimatch
    """
    all_mm_scores = []
    for traj in model_trajs:
        img_name = traj['name']
        task = traj['task']
        gt_trajs = list(
            filter(lambda x: x['name'] == img_name and x['task'] == task,
                   human_trajs))
        all_mm_scores.append((task,
                              np.mean([
                                  multimatch(traj, gt_traj, (im_w, im_h))[:4]
                                  for gt_traj in gt_trajs
                              ],
                                      axis=0)))

    if tasks is not None:
        mm_tasks = {}
        for task in tasks:
            mm = np.array([x[1] for x in all_mm_scores if x[0] == task])
            mm_tasks[task] = np.mean(mm, axis=0)
        return mm_tasks
    else:
        return np.mean([x[1] for x in all_mm_scores], axis=0)


def scanpath2clusters(meanshift, scanpath):
    string = []
    xs = scanpath['X']
    ys = scanpath['Y']
    for i in range(len(xs)):
        symbol = meanshift.predict([[xs[i], ys[i]]])[0]
        string.append(symbol)
    return string


def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0


def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))


# compute sequence score
def compute_SS(preds, clusters, truncate, truncate_gt, reduce='mean'):
    results = []
    for scanpath in preds:
        is_fv = scanpath['condition'] == 'freeview'
        if is_fv:
            key = 'test-{}-{}'.format(scanpath['condition'], scanpath['name'].split('.')[0])
        else:
            key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                         scanpath['name'].split('.')[0])
        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']

        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        if len(strings) == 0:
            continue
        for gt in strings:
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                if truncate_gt:
                    gt = gt[:truncate] if len(gt) > truncate else gt
                score = nw_matching(pred, gt)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        if not is_fv:
            result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results


def get_seq_score(preds, clusters, max_step, truncate_gt=False, tasks=None):
    results = compute_SS(preds, clusters, max_step, truncate_gt)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))


def scanpath2categories(seg_map, scanpath):
    string = []
    xs = scanpath['X']
    ys = scanpath['Y']
    for x, y in zip(xs, ys):
        symbol = str(int(seg_map[int(y), int(x)]))
        string.append(symbol)
    return string


# compute semantic sequence score
def compute_SSS(preds,
                fixations,
                truncate,
                segmentation_map_dir,
                truncate_gt,
                reduce='mean'):
    results = []
    for scanpath in preds:
        is_fv = scanpath['condition'] == 'freeview'
        if is_fv:
            key = 'test-{}-{}'.format(scanpath['condition'], scanpath['name'].split('.')[0])
        else:
            key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                         scanpath['name'].split('.')[0])
        strings = fixations[key]
        with gzip.GzipFile(
                join(segmentation_map_dir, scanpath['name'][:-3] + 'npy.gz'),
                "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        pred = scanpath2categories(segmentation_map, scanpath)
        scores = []
        human_scores = []
        for gt in strings:
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                if truncate_gt:
                    gt = gt[:truncate] if len(gt) > truncate else gt
                score = nw_matching(pred, gt)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        if not is_fv:
            result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results


def get_semantic_seq_score(preds,
                           fixations,
                           max_step,
                           segmentation_map_dir,
                           truncate_gt=False,
                           tasks=None):
    results = compute_SSS(preds, fixations, max_step, segmentation_map_dir,
                          truncate_gt)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))


def scanpath_ratio(traj, bbox):
    X1, Y1 = traj['X'][:-1], traj['Y'][:-1]
    X2, Y2 = traj['X'][1:], traj['Y'][1:]
    traj_dist = np.sum(np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2))
    cx, cy = traj['X'][0], traj['Y'][0]
    tx, ty = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
    target_dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)
    if traj_dist == 0:
        print("error traj", traj)
    return min(target_dist / traj_dist, 1.0)


def compute_avgSPRatio(trajs, target_annos, max_step, tasks=None):

    all_sp_ratios = []
    for traj in trajs:
        key = traj['task'] + '_' + traj['name']
        bbox = target_annos[key]
        num_step = utils.get_num_step2target(traj['X'], traj['Y'], bbox)
        if num_step > max_step + 1:  # skip failed scanpaths
            continue
        sp = {'X': traj['X'][:num_step], 'Y': traj['Y'][:num_step]}
        if len(sp['X']) == 1:  # skip single-step scanpaths
            continue
        all_sp_ratios.append((traj['task'], scanpath_ratio(sp, bbox)))

    if tasks is not None:
        avg_sp_ratios = {}
        for task in tasks:
            sp_ratios = [x[1] for x in all_sp_ratios if x[0] == task]
            avg_sp_ratios[task] = np.mean(sp_ratios)
        return avg_sp_ratios
    else:
        return np.mean([x[1] for x in all_sp_ratios])


def compute_cdf_auc(cdf):
    if isinstance(cdf, dict):
        auc = {}
        for k, v in cdf.items():
            auc[k] = v[0] + v[-1] + np.sum(v[1:-1])
        return auc
    else:
        return cdf[0] + cdf[-1] + np.sum(cdf[1:-1])


def compute_prob_mismatch(cdf, human_mean_cdf):
    if isinstance(cdf, dict):
        return dict(
            zip(
                cdf.keys(),
                np.sum(np.abs(np.array(list(cdf.values())) - human_mean_cdf),
                       axis=1)))
    else:
        return np.sum(np.abs(cdf - human_mean_cdf))


def smooth_action_map(action_maps):
    for i in range(len(action_maps)):
        action_maps[i] = filters.gaussian_filter(action_maps[i], sigma=1)
    return action_maps


def compute_sequence_KL(probs, gt_probs, patch_num):
    # blur action map
    smooth_probs = smooth_action_map(
        probs.view(-1, patch_num[1], patch_num[0]).cpu().numpy())
    smooth_probs = torch.from_numpy(smooth_probs).view(smooth_probs.shape[0],
                                                       -1).to(gt_probs.device)
    log_probs = torch.log(smooth_probs)
    kl_div = torch.nn.functional.kl_div(log_probs,
                                        gt_probs,
                                        reduction='batchmean')
    return kl_div


def info_gain(predicted_probs, gt_fixs, base_probs, eps=2.2204e-16):
    fired_probs = predicted_probs[gt_fixs[:, 1], gt_fixs[:, 0]]
    fired_base_probs = base_probs[gt_fixs[:, 1], gt_fixs[:, 0]]
    IG = np.sum(np.log2(fired_probs + eps) - np.log2(fired_base_probs + eps))
    return IG


def CC(saliency_map_1, saliency_map_2):
    def normalize(saliency_map):
        saliency_map -= saliency_map.mean()
        std = saliency_map.std()

        if std:
            saliency_map /= std

        return saliency_map, std == 0

    smap1, constant1 = normalize(saliency_map_1.copy())
    smap2, constant2 = normalize(saliency_map_2.copy())

    if constant1 and not constant2:
        return 0.0
    else:
        return np.corrcoef(smap1.flatten(), smap2.flatten())[0, 1]


def NSS(saliency_map, gt_fixs):
    xs, ys = gt_fixs[:, 0], gt_fixs[:, 1]

    mean = saliency_map.mean()
    std = saliency_map.std()

    value = saliency_map[ys, xs].copy()
    value -= mean

    if std:
        value /= std

    return value.sum()


def compute_spatial_metrics_by_step(predicted_trajs,
                                    gt_scanpaths,
                                    im_w,
                                    im_h,
                                    prior_maps,
                                    end_step=1):
    sample_ids = np.unique(
        [traj['task'] + '_' + traj['name'] for traj in predicted_trajs])

    avg_info_gain = 0
    num_fixs = 0
    cc = 0
    nss = 0
    for sample_id in sample_ids:
        task, image = sample_id.split('_')
        trajs = list(
            filter(lambda x: x['task'] == task and x['name'] == image,
                   predicted_trajs))
        assert len(trajs) > 0, 'empty trajs.'

        # removing the predifined first fixation
        if end_step > 1:
            Xs = np.concatenate([traj['X'][1:end_step] for traj in trajs])
            Ys = np.concatenate([traj['Y'][1:end_step] for traj in trajs])
        else:
            Xs = np.concatenate([traj['X'][1:] for traj in trajs])
            Ys = np.concatenate([traj['Y'][1:] for traj in trajs])
        fixs = np.stack([Xs, Ys]).T.astype(np.int32)
        pred_smap = utils.convert_fixations_to_map(fixs,
                                                   im_w,
                                                   im_h,
                                                   smooth=True)

        gt_trajs = list(
            filter(lambda x: x['task'] == task and x['name'] == image,
                   gt_scanpaths))
        assert len(gt_trajs) > 0, 'empty trajs.'
        if end_step > 1:
            Xs = np.concatenate([traj['X'][1:end_step] for traj in gt_trajs])
            Ys = np.concatenate([traj['Y'][1:end_step] for traj in gt_trajs])
        else:
            Xs = np.concatenate([traj['X'][1:] for traj in gt_trajs])
            Ys = np.concatenate([traj['Y'][1:] for traj in gt_trajs])
        gt_fixs = np.stack([Xs, Ys]).T.astype(np.int32)
        gt_smap = utils.convert_fixations_to_map(gt_fixs,
                                                 im_w,
                                                 im_h,
                                                 smooth=True)

        avg_info_gain += info_gain(pred_smap, gt_fixs, prior_maps[task])
        num_fixs += len(gt_fixs)

        cc += CC(pred_smap, gt_smap)
        nss += NSS(pred_smap, gt_fixs)

    return avg_info_gain / num_fixs, cc / len(sample_ids), nss / num_fixs


# Conditional saliency scores
def compute_info_gain(predicted_probs, gt_fixs, base_probs, eps=2.2204e-16):
    fired_probs = predicted_probs[torch.arange(gt_fixs.size(0)
                                               ), gt_fixs[:, 1], gt_fixs[:, 0]]
    fired_base_probs = base_probs[torch.arange(gt_fixs.size(0)
                                               ), gt_fixs[:, 1], gt_fixs[:, 0]]
    IG = torch.sum(
        torch.log2(fired_probs + eps) - torch.log2(fired_base_probs + eps))
    return IG


def compute_NSS(saliency_map, gt_fixs):
    mean = saliency_map.view(gt_fixs.size(0), -1).mean(dim=1)
    std = saliency_map.view(gt_fixs.size(0), -1).std(dim=1)
    std[std == 0] = 1  # avoid division by 0

    value = saliency_map[torch.arange(gt_fixs.size(0)
                                      ), gt_fixs[:, 1], gt_fixs[:, 0]]
    value -= mean
    value /= std

    return value.sum()

def compute_cAUC(s_map, gt_next_fixs):
    """Compute AUC_Judd metric for saliency maps.
       
    This is equivalent to compute the percentile of the saliency of ground-truth 
    fixation in the predicted saliency map.

    Args:
       s_map: [B, H, W] tensor
       gt_next_fixs: [B, 2] tensor
    """
    # thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = s_map[torch.arange(len(gt_next_fixs)), 
                       gt_next_fixs[:, 1], 
                       gt_next_fixs[:, 0]]

    bs = len(gt_next_fixs)

    area = []
    area.append(torch.zeros(bs, 2))
        
    # In the salience map, keep only those pixels with values above threshold
    temp = torch.zeros_like(s_map)
    temp[s_map>=thresholds.view(bs, 1, 1)] = 1.0
    temp = temp.view(bs, -1)
    
    # For each image, three is only one positive
    tp = torch.ones(bs)
    fp = (temp.sum(-1) - 1)/(temp.size(-1) - 1)
    area.append(torch.stack([tp, fp.cpu()], dim=1))
    area.append(torch.ones(bs, 2))
    area = torch.stack(area, dim=1)

    return torch.trapz(area[:, :, 0], area[:, :, 1]).sum()