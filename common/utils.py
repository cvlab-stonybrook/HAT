import numpy as np
import torch
import math
import torch.nn.functional as F
from copy import copy
from torch.distributions import Categorical
import scipy.ndimage as filters
import warnings
import os
import re, cv2
from shutil import copyfile
warnings.filterwarnings("ignore", category=UserWarning)


def get_foveal_weights(fixation_batch,
                       width,
                       height,
                       sigma=0.248,
                       p=7.5,
                       k=1.5,
                       alpha=1.25):
    """
    This function generate foveated image in batch on GPU
    
    fixation_batch: normalized fixation tensor of shape (batch_size, 
      fix_num, 2) 
    """
    assert fixation_batch.size(-1) == 2, 'Wrong input shape!'
    assert fixation_batch.max() <= 1, 'Fixation has to be normalized!'
    prNum = 5

    batch_size = fixation_batch.size(0)
    fix_num = fixation_batch.size(1)
    device = fixation_batch.device

    # Map fixations to coordinate space
    fixation_batch = fixation_batch * torch.tensor([width, height]).to(device)

    x = torch.arange(0, width, device=device, dtype=torch.float)
    y = torch.arange(0, height, device=device, dtype=torch.float)
    y2d, x2d = torch.meshgrid([y, x])
    h, w = x2d.size()

    x2d = x2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)
    y2d = y2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)

    # fixation patch index to fixation pixel coordinates
    xc = fixation_batch[:, :, 0]
    yc = fixation_batch[:, :, 1]

    xc2d = xc.view(batch_size, fix_num, 1, 1).expand_as(x2d)
    yc2d = yc.view(batch_size, fix_num, 1, 1).expand_as(y2d)

    theta = torch.sqrt((x2d - xc2d)**2 + (y2d - yc2d)**2) / p
    theta, _ = torch.min(theta, dim=1)
    R = alpha / (theta + alpha)

    Ts = torch.zeros((batch_size, prNum, height, width), device=device)
    for i in range(prNum - 1):
        Ts[:, i] = torch.exp(-((2**(i - 2)) * R / sigma)**2 * k)

    # omega
    omega = torch.zeros(prNum)
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k)
    omega[:-1] = torch.sqrt(math.log(2) / k) / (2**torch.arange(
        -2, prNum // 2, dtype=torch.float32, device=device)) * sigma
    omega[omega > 1] = 1

    # layer index
    layer_ind = torch.zeros_like(R, device=device)
    for i in range(1, prNum):
        ind = (R >= omega[i]) * (R <= omega[i - 1])
        layer_ind[ind] = i

    # Bs
    Bs = (0.5 - Ts[:, 1:]) / (Ts[:, :-1] - Ts[:, 1:])

    # M
    Ms = torch.zeros((batch_size, prNum, height, width), device=device)
    for i in range(prNum):
        ind = layer_ind == i
        if torch.sum(ind) > 0:
            if i == 0:
                Ms[:, i][ind] = 1
            else:
                Ms[:, i][ind] = 1 - Bs[:, i - 1][ind]

        ind = layer_ind - 1 == i
        if torch.sum(ind) > 0:
            Ms[:, i][ind] = Bs[:, i][ind]

    return Ms


def cutFixOnTarget(trajs, target_annos):
    task_names = np.unique([traj['task'] for traj in trajs])
    if 'condition' in trajs[0].keys():
        trajs = list(filter(lambda x: x['condition'] == 'present', trajs))
    if len(trajs) == 0:
        return
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['name']
            bbox = target_annos[key]
            traj_len = get_num_step2target(traj['X'], traj['Y'], bbox)
            num_steps_task[i] = traj_len
            traj['X'] = traj['X'][:traj_len]
            traj['Y'] = traj['Y'][:traj_len]


def pos_to_action(center_x, center_y, patch_size, patch_num):
    x = center_x // patch_size[0]
    y = center_y // patch_size[1]

    return int(patch_num[0] * y + x)


def action_to_pos(acts, patch_size, patch_num):
    patch_y = acts // patch_num[0]
    patch_x = acts % patch_num[0]

    pixel_x = patch_x * patch_size[0] + patch_size[0] / 2
    pixel_y = patch_y * patch_size[1] + patch_size[1] / 2
    return pixel_x, pixel_y


def select_action(obs,
                  policy,
                  sample_action,
                  action_mask=None,
                  softmask=False,
                  eps=1e-12,
                  has_stop=False):
    probs, values = policy(*obs)
    if sample_action:
        m = Categorical(probs)
        if action_mask is not None:
            # prevent sample previous actions by re-normalizing probs
            probs_new = probs.clone().detach()
            if softmask:
                probs_new = probs_new * action_mask
            else:
                if has_stop:
                    probs_new[:, :-1][action_mask] = eps
                else:
                    probs_new[action_mask] = eps

            probs_new /= probs_new.sum(dim=1).view(probs_new.size(0), 1)

            m_new = Categorical(probs_new)
            actions = m_new.sample()
        else:
            actions = m.sample()
        log_probs = m.log_prob(actions)
        return actions.view(-1), log_probs, values.view(-1), probs
    else:
        probs_new = probs.clone().detach()
        if has_stop:
            probs_new[:, :-1][action_mask.view(probs_new.size(0), -1)] = 0
        else:
            probs_new[action_mask.view(probs_new.size(0), -1)] = 0
        actions = torch.argmax(probs_new, dim=1)
        return actions.view(-1), None, None, None


def collect_trajs(env,
                  policy,
                  patch_num,
                  max_traj_length,
                  is_eval=False,
                  sample_action=True,
                  is_zero_shot=False):

    rewards = []
    obs_fov = env.observe()
    is_composite_state = isinstance(obs_fov, tuple)

    def pack_model_inputs(obs_fov, env):
        if is_composite_state:
            inputs = [*obs_fov, env.task_ids]
        else:
            inputs = [obs_fov, env.task_ids]
        if is_zero_shot:
            inputs.append(env.hr_feats)
        return inputs

    inputs = pack_model_inputs(obs_fov, env)
    act, log_prob, value, prob = select_action(inputs,
                                               policy,
                                               sample_action,
                                               action_mask=env.action_mask,
                                               has_stop=env.pa.has_stop)
    status = [env.status]
    values = [value]
    log_probs = [log_prob]
    SASPs = []

    i = 0
    if is_eval:
        actions = []
        while i < max_traj_length:
            new_obs_fov, curr_status = env.step(act)
            status.append(curr_status)
            actions.append(act)
            obs_fov = new_obs_fov
            inputs = pack_model_inputs(obs_fov, env)
            act, log_prob, value, prob_new = select_action(
                inputs,
                policy,
                sample_action,
                action_mask=env.action_mask,
                has_stop=env.pa.has_stop)
            i = i + 1

        status = torch.stack(status[1:])
        actions = torch.stack(actions)

        bs = len(env.img_names)
        trajs = []
        for i in range(bs):
            ind = (status[:, i] == 1).to(torch.int8).argmax().item() + 1
            if status[:, i].sum() == 0:
                ind = status.size(0)
            trajs.append({'actions': actions[:ind, i]})

    else:
        IORs = []
        IORs.append(
            env.action_mask.to(dtype=torch.float).view(env.batch_size, 1,
                                                       patch_num[1], -1))
        while i < max_traj_length and env.status.min() < 1:
            new_obs_fov, curr_status = env.step(act)

            status.append(curr_status)
            SASPs.append((obs_fov, act, new_obs_fov))
            obs_fov = new_obs_fov

            IORs.append(
                env.action_mask.to(dtype=torch.float).view(
                    env.batch_size, 1, patch_num[1], -1))
            inputs = pack_model_inputs(obs_fov, env)
            act, log_prob, value, prob_new = select_action(
                inputs,
                policy,
                sample_action,
                action_mask=env.action_mask,
                has_stop=env.pa.has_stop)
            values.append(value)
            log_probs.append(log_prob)

            # place holder, reward assigned after collection is done
            rewards.append(torch.zeros(env.batch_size))

            i = i + 1

        if is_composite_state:
            num_state_comps = len(SASPs[0][0])
            S = [
                torch.stack([sasp[0][i] for sasp in SASPs])
                for i in range(num_state_comps)
            ]
        else:
            S = torch.stack([sasp[0] for sasp in SASPs])
        A = torch.stack([sasp[1] for sasp in SASPs])
        V = torch.stack(values)
        R = torch.stack(rewards)
        LogP = torch.stack(log_probs[:-1])
        status = torch.stack(status[1:])

        bs = len(env.img_names)
        trajs = []

        for i in range(bs):
            ind = (status[:, i] == 1).to(torch.int8).argmax().item() + 1
            if status[:, i].sum() == 0:
                ind = status.size(0)
            trajs.append({
                'curr_states':
                [s[:ind, i] for s in S] if is_composite_state else S[:ind, i],
                'actions':
                A[:ind, i],
                'values':
                V[:ind + 1, i],
                'log_probs':
                LogP[:ind, i],
                'rewards':
                R[:ind, i],
                'task_id':
                env.task_ids[i].repeat(ind),
                'img_name': [env.img_names[i]] * ind,
                'length':
                ind,
                'hr_feats':
                torch.stack([env.hr_feats[i]] * ind) if is_zero_shot else None
            })

    return trajs


def compute_return_advantage(rewards, values, gamma, mtd='CRITIC', tau=0.96):
    device = rewards.device
    acc_reward = torch.zeros_like(rewards, dtype=torch.float, device=device)
    acc_reward[-1] = rewards[-1]
    for i in reversed(range(acc_reward.size(0) - 1)):
        acc_reward[i] = rewards[i] + gamma * acc_reward[i + 1]

    # compute advantages
    if mtd == 'MC':  # Monte-Carlo estimation
        advs = acc_reward - values[:-1]
    elif mtd == 'CRITIC':  # critic estimation
        advs = rewards + gamma * values[1:] - values[:-1]
    elif mtd == 'GAE':  # generalized advantage estimation
        delta = rewards + gamma * values[1:] - values[:-1]
        adv = torch.zeros_like(delta, dtype=torch.float, device=device)
        adv[-1] = delta[-1]
        for i in reversed(range(delta.size(0) - 1)):
            adv[i] = delta[i] + gamma * tau * adv[i + 1]
    else:
        raise NotImplementedError

    return acc_reward.squeeze(), advs.squeeze()


def process_trajs(trajs, gamma, mtd='CRITIC', tau=0.96):
    # compute discounted cummulative reward
    device = trajs[0]['log_probs'].device
    avg_return = 0
    for traj in trajs:

        acc_reward = torch.zeros_like(traj['rewards'],
                                      dtype=torch.float,
                                      device=device)
        acc_reward[-1] = traj['rewards'][-1]
        for i in reversed(range(acc_reward.size(0) - 1)):
            acc_reward[i] = traj['rewards'][i] + gamma * acc_reward[i + 1]

        traj['acc_rewards'] = acc_reward
        avg_return += acc_reward[0]

        values = traj['values']
        # compute advantages
        if mtd == 'MC':  # Monte-Carlo estimation
            traj['advantages'] = traj['acc_rewards'] - values[:-1]

        elif mtd == 'CRITIC':  # critic estimation
            traj['advantages'] = traj[
                'rewards'] + gamma * values[1:] - values[:-1]

        elif mtd == 'GAE':  # generalized advantage estimation
            delta = traj['rewards'] + gamma * values[1:] - values[:-1]
            adv = torch.zeros_like(delta, dtype=torch.float, device=device)
            adv[-1] = delta[-1]
            for i in reversed(range(delta.size(0) - 1)):
                adv[i] = delta[i] + gamma * tau * adv[i + 1]
            traj['advantages'] = adv
        else:
            raise NotImplementedError

    return avg_return / len(trajs)


def get_num_step2target(X, Y, bbox):
    X, Y = np.array(X), np.array(Y)
    on_target_X = np.logical_and(X > bbox[0], X < bbox[0] + bbox[2])
    on_target_Y = np.logical_and(Y > bbox[1], Y < bbox[1] + bbox[3])
    on_target = np.logical_and(on_target_X, on_target_Y)
    if np.sum(on_target) > 0:
        first_on_target_idx = np.argmax(on_target)
        return first_on_target_idx + 1
    else:
        return 1000  # some big enough number


def get_CDF(num_steps, max_step):
    cdf = np.zeros(max_step)
    total = float(len(num_steps))
    for i in range(1, max_step + 1):
        cdf[i - 1] = np.sum(num_steps <= i) / total
    return cdf


def get_num_steps(trajs, target_annos, task_names):
    num_steps = {}
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['name']
            bbox = target_annos[key]
            step_num = get_num_step2target(np.array(traj['X']),
                                           np.array(traj['Y']), bbox)
            num_steps_task[i] = step_num
            traj['X'] = traj['X'][:step_num]
            traj['Y'] = traj['Y'][:step_num]
        num_steps[task] = num_steps_task
    return num_steps


def get_mean_cdf(num_steps, task_names, max_step):
    cdf_tasks = []
    for task in task_names:
        cdf_tasks.append(get_CDF(num_steps[task], max_step))
    return cdf_tasks


def compute_search_cdf(scanpaths, annos, max_step, return_by_task=False):
    # compute search CDF
    task_names = np.unique([traj['task'] for traj in scanpaths])
    num_steps = get_num_steps(scanpaths, annos, task_names)
    cdf_tasks = get_mean_cdf(num_steps, task_names, max_step + 1)
    if return_by_task:
        return dict(zip(task_names, cdf_tasks))
    else:
        mean_cdf = np.mean(cdf_tasks, axis=0)
        std_cdf = np.std(cdf_tasks, axis=0)
        return mean_cdf, std_cdf


def calc_overlap_ratio(bbox, patch_size, patch_num):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    patch_area = float(patch_size[0] * patch_size[1])
    aoi_ratio = np.zeros((1, patch_num[1], patch_num[0]), dtype=np.float32)

    tl_x, tl_y = bbox[0], bbox[1]
    br_x, br_y = bbox[0] + bbox[2], bbox[1] + bbox[3]
    lx, ux = tl_x // patch_size[0], br_x // patch_size[0]
    ly, uy = tl_y // patch_size[1], br_y // patch_size[1]

    for x in range(lx, ux + 1):
        for y in range(ly, uy + 1):
            patch_tlx, patch_tly = x * patch_size[0], y * patch_size[1]
            patch_brx, patch_bry = patch_tlx + patch_size[
                0], patch_tly + patch_size[1]

            aoi_tlx = tl_x if patch_tlx < tl_x else patch_tlx
            aoi_tly = tl_y if patch_tly < tl_y else patch_tly
            aoi_brx = br_x if patch_brx > br_x else patch_brx
            aoi_bry = br_y if patch_bry > br_y else patch_bry

            aoi_ratio[0, y, x] = max((aoi_brx - aoi_tlx), 0) * max(
                (aoi_bry - aoi_tly), 0) / float(patch_area)

    return aoi_ratio


def get_center_keypoint_map(bbox, patch_num, box_size_dependent=True, normalize=True):
    xc, yc = np.round(bbox[0] + bbox[2] / 2), np.round(bbox[1] + bbox[3] / 2)
    if box_size_dependent:
        sigma = np.sqrt(bbox[2] ** 2 + bbox[3] ** 2) / 8
        target_map = np.zeros((320, 512), dtype=np.float32)
        target_map[int(yc), int(xc)] = 1
        target_map = filters.gaussian_filter(target_map, sigma=sigma)
        if patch_num[0] < 320:
            target_map = F.interpolate(
                torch.from_numpy(target_map).unsqueeze(0).unsqueeze(0),
                size=patch_num, mode='bilinear')
    else:
        target_map = np.zeros(patch_num, dtype=np.float32)
        target_map[int(yc//16), int(xc//16)] = 1
        target_map = filters.gaussian_filter(target_map, sigma=1)
        target_map = torch.from_numpy(target_map)

    if normalize:
        target_map /= target_map.sum()
    return target_map


def foveal2mask(x, y, r, h, w):
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    mask = dist <= r
    return mask.astype(np.float32)


def multi_hot_coding(bbox, patch_size, patch_num, thresh=0):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    aoi_ratio = calc_overlap_ratio(bbox, patch_size, patch_num)
    hot_ind = aoi_ratio > thresh
    while hot_ind.sum() == 0:
        thresh *= 0.5
        hot_ind = aoi_ratio > thresh

    aoi_ratio[hot_ind] = 1
    aoi_ratio[np.logical_not(hot_ind)] = 0

    return aoi_ratio[0]


def actions2scanpaths(actions_all, patch_num):
    # convert actions to scanpaths
    scanpaths = []
    for traj in actions_all:
        task_name, img_name, condition, actions = traj
        actions = actions.to(dtype=torch.float32)
        if actions[-1] == patch_num[0] * patch_num[1]:
            actions = actions[:-1]  # remove stopping action
        py = (actions // patch_num[0]) / float(patch_num[1])
        px = (actions % patch_num[0]) / float(patch_num[0])
        fixs = torch.stack([px, py])
        fixs = np.concatenate([np.array([[0.5], [0.5]]),
                               fixs.cpu().numpy()],
                              axis=1)
        scanpaths.append({
            'X': fixs[0] * 512,
            'Y': fixs[1] * 320,
            'name': img_name,
            'task': task_name,
            'condition': condition
        })
    return scanpaths


def preprocess_fixations(trajs,
                         patch_size,
                         patch_num,
                         im_h,
                         im_w,
                         truncate_num=-1,
                         need_label=True,
                         has_stop=False,
                         sample_scanpath=False,
                         min_traj_length_percentage=0,
                         discretize_fix=True,
                         remove_return_fixations=False,
                         is_coco_dataset=True):
    fix_labels = []
    if len(trajs) == 0:
        return fix_labels
    
    has_spid = 'scanpath_id' in trajs[0].keys()
    for traj in trajs:
        if is_coco_dataset:  # For COCO-Search/Freeview datasets, force the initial fixation to be the center
            traj['X'][0], traj['Y'][0] = im_w // 2, im_h // 2
        discrete_label = pos_to_action(
            traj['X'][0], traj['Y'][0], patch_size, patch_num)
        if discretize_fix:
            label = discrete_label
            tar_x, tar_y = action_to_pos(label, patch_size, patch_num)
            fixs = [(tar_x, tar_y)]
        else:
            label = pos_to_action(
                traj['X'][0], traj['Y'][0], [1, 1], [im_w, im_h])
            fixs = [(traj['X'][0], traj['Y'][0])]
        label_his = [discrete_label]
        if truncate_num < 1:
            traj_len = len(traj['X'])
        else:
            traj_len = min(truncate_num, len(traj['X']))

        min_traj_length = int(min_traj_length_percentage * traj_len)

        if not is_coco_dataset and not sample_scanpath:
            # Predict the initial fixation for datasets other than COCO-Search/Freeview
            if has_spid:
                fix_label = [traj['name'], traj['task'], traj['condition'],
                             [], label, False, traj['subject'],
                             traj['scanpath_id'], 0]
            else:
                fix_label = [traj['name'], traj['task'], traj['condition'],
                             [], label, False, traj['subject'], 0]
            fix_labels.append(fix_label)

        for i in range(1, traj_len):
            if traj['X'][i] >= im_w or traj['Y'][i] >= im_h or traj['X'][i] < 0 or traj['Y'][i] < 0:
                # Remove out-of-bound fixations
                continue
            discrete_label = pos_to_action(
                traj['X'][i], traj['Y'][i], patch_size, patch_num)
            if discretize_fix:
                label = discrete_label
            else:
                label = pos_to_action(
                    traj['X'][i], traj['Y'][i], [1, 1], [im_w, im_h])
            # remove returning fixations (enforce inhibition of return)
            if remove_return_fixations and discrete_label in label_his:
                continue
            label_his.append(discrete_label)
            if has_spid:
                fix_label = [traj['name'], traj['task'], traj['condition'],
                             copy(fixs), label, False, traj['subject'],
                             traj['scanpath_id'], np.sum(traj['T'][:i])]
            else:
                fix_label = [traj['name'], traj['task'], traj['condition'],
                             copy(fixs), label, False, traj['subject'],
                             np.sum(traj['T'][:i])]

            # discretize fixations
            if discretize_fix:
                tar_x, tar_y = action_to_pos(label, patch_size, patch_num)
                fixs.append((tar_x, tar_y))
            else:
                fixs.append((traj['X'][i], traj['Y'][i]))
            
            if (not sample_scanpath) and i >= min_traj_length:
                fix_labels.append(fix_label)

        if has_stop or sample_scanpath:
            # append the stopping action at the end of the scanapth
            if has_spid:
                stop_label = [traj['name'], traj['task'], traj['condition'],
                              copy(fixs), patch_num[0] * patch_num[1], True,
                              traj['subject'], traj['scanpath_id'],
                              np.sum(traj['T'])]
            else:
                stop_label = [traj['name'], traj['task'], traj['condition'],
                              copy(fixs), patch_num[0] * patch_num[1], True,
                              traj['subject'], np.sum(traj['T'])]
            fix_labels.append(stop_label)

    return fix_labels


def _file_at_step(step, name):
    return "save_{}_{}k{}.pkg".format(name, int(step // 1000),
                                      int(step % 1000))


def _file_best(name):
    return "trained_{}.pkg".format(name)


def save(global_step,
         model,
         optim,
         name,
         pkg_dir="",
         is_best=False,
         max_checkpoints=None):
    if optim is None:
        raise ValueError("cannot save without optimzier")
    state = {
        "global_step":
        global_step,
        # DataParallel wrap model in attr `module`.
        "model":
        model.module.state_dict()
        if hasattr(model, "module") else model.state_dict(),
        "optim":
        optim.state_dict(),
    }
    save_path = os.path.join(pkg_dir, _file_at_step(global_step, name))
    best_path = os.path.join(pkg_dir, _file_best(name))
    torch.save(state, save_path)
    print("[Checkpoint]: save to {} successfully".format(save_path))

    if is_best:
        copyfile(save_path, best_path)
    if max_checkpoints is not None:
        history = []
        for file_name in os.listdir(pkg_dir):
            if re.search("save_{}_\d*k\d*\.pkg".format(name), file_name):
                digits = file_name.replace("save_{}_".format(name),
                                           "").replace(".pkg", "").split("k")
                number = int(digits[0]) * 1000 + int(digits[1])
                history.append(number)
        history.sort()
        while len(history) > max_checkpoints:
            path = os.path.join(pkg_dir, _file_at_step(history[0], name))
            print("[Checkpoint]: remove {} to keep {} checkpoints".format(
                path, max_checkpoints))
            if os.path.exists(path):
                os.remove(path)
            history.pop(0)


def load(step_or_path, model, name, optim=None, pkg_dir="", device=None):
    step = step_or_path
    save_path = None
    if isinstance(step, int):
        save_path = os.path.join(pkg_dir, _file_at_step(step, name))
    if isinstance(step, str):
        if pkg_dir is not None:
            if step == "best":
                save_path = os.path.join(pkg_dir, _file_best(name))
            else:
                save_path = os.path.join(pkg_dir, step)
        else:
            save_path = step
    if save_path is not None and not os.path.exists(save_path):
        print("[Checkpoint]: Failed to find {}".format(save_path))
        return
    if save_path is None:
        print("[Checkpoint]: Cannot load the checkpoint")
        return

    # begin to load
    state = torch.load(save_path, map_location=device)
    global_step = state["global_step"]
    model.load_state_dict(state["model"])
    if optim is not None:
        optim.load_state_dict(state["optim"])

    print("[Checkpoint]: Load {} successfully".format(save_path))
    return global_step


def genGaussiankernel(width, sigma):
    x = np.arange(-int(width / 2), int(width / 2) + 1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d**2 + y2d**2) / (2 * sigma**2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d


def pyramid(im, sigma=1, prNum=6, transform=None):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]

    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)

    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width / 2), int(height / 2)))
        pyramids.append(G)

    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            curr_im = cv2.resize(curr_im,
                                 (curr_im.shape[1] * 2, curr_im.shape[0] * 2))
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids


def foveat_img(im, fixs, As=None):
    sigma = 0.248
    prNum = 6
    if As is None:
        As = pyramid(im, sigma, prNum)
        height, width, _ = im.shape
    else:
        height, width, _ = As[0].shape

    # compute coef
    p = 7.5  # 16
    k = 1.5  # 1.02
    alpha = 1.5

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0])**2 + (y2d - fixs[0][1])**2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta,
                           np.sqrt((x2d - fix[0])**2 + (y2d - fix[1])**2) / p)
    R = alpha / (theta + alpha)

    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2**(i - 3)) * R / sigma)**2 * k))
    Ts.append(np.zeros_like(theta))

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i - 1] = np.sqrt(np.log(2) / k) / (2**(i - 3)) * sigma

    omega[omega > 1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i - 1] - Ts[i] + 1e-5))

    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))
    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i - 1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])

    im_fov = im_fov.astype(np.uint8)
    return im_fov, Ms


def real_foveation_batch(As, fixation_batch, pa):
    """
    This function generate foveated image in batch on GPU
    
    **As**: batch of image pyrimaids of the shape (batch_size, 
    pyrNum, channel, height, width).
    
    **fixation_batch**: (batch_size, fix_num, 2) tensor in (x, y)
    """
    sigma = 0.248
    prNum = 6
    width = pa.im_w
    height = pa.im_h
    patch_size = pa.patch_size
    device = As.device

    prNum = As.size()[1]
    batch_size = As.size()[0]
    fix_num = fixation_batch.size(1)

    # compute coef
    p = 7.5
    k = 1.5
    alpha = 2.5

    x = torch.arange(0, width, device=device, dtype=torch.float)
    y = torch.arange(0, height, device=device, dtype=torch.float)
    y2d, x2d = torch.meshgrid([y, x])
    h, w = x2d.size()

    x2d = x2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)
    y2d = y2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)

    # fixation patch index to fixation pixel coordinates
    xc = fixation_batch[:, :, 0] * patch_size[0] + patch_size[0] / 2
    yc = fixation_batch[:, :, 1] * patch_size[1] + patch_size[1] / 2

    xc2d = xc.view(batch_size, fix_num, 1, 1).expand_as(x2d)
    yc2d = yc.view(batch_size, fix_num, 1, 1).expand_as(y2d)

    theta = torch.sqrt((x2d - xc2d)**2 + (y2d - yc2d)**2) / p
    theta, _ = torch.min(theta, dim=1)
    R = alpha / (theta + alpha)

    Ts = torch.zeros((batch_size, 6, height, width), device=device)
    for i in range(prNum - 1):
        Ts[:, i] = torch.exp(-((2**(i - 2)) * R / sigma)**2 * k)

    # omega
    omega = np.zeros(prNum)
    omega[:-1] = math.sqrt(math.log(2) / k) / (2**np.arange(
        -2, 3, dtype=float)) * sigma
    omega[omega > 1] = 1

    # layer index
    layer_ind = torch.zeros_like(R, device=device)
    for i in range(1, prNum):
        ind = (R >= omega[i]) * (R <= omega[i - 1])
        layer_ind[ind] = i

    # Bs
    Bs = (0.5 - Ts[:, 1:]) / (Ts[:, :-1] - Ts[:, 1:])

    # M
    Ms = torch.zeros((batch_size, prNum, height, width), device=device)
    for i in range(prNum):
        ind = layer_ind == i
        if torch.sum(ind) > 0:
            if i == 0:
                Ms[:, i][ind] = 1
            else:
                Ms[:, i][ind] = 1 - Bs[:, i - 1][ind]

        ind = layer_ind - 1 == i
        if torch.sum(ind) > 0:
            Ms[:, i][ind] = Bs[:, i][ind]

    # generate periphery image
    Ms = Ms.unsqueeze(2).expand(-1, -1, 3, -1, -1)
    im_fov_batch = torch.sum(Ms * As, dim=1)

    return im_fov_batch, R


# Convert discrete fixation data to continuous density map
def convert_fixations_to_map(fixs,
                             width,
                             height,
                             return_distribution=True,
                             smooth=True,
                             visual_angle=16):
    assert len(fixs) > 0, 'Empty fixation list!'

    fmap = np.zeros((height, width))
    for i in range(len(fixs)):
        x, y = fixs[i][0], fixs[i][1]
        fmap[y, x] += 1

    if smooth:
        fmap = filters.gaussian_filter(fmap, sigma=visual_angle)

    if return_distribution:
        fmap /= fmap.sum()

    return fmap


def get_prior_maps(gt_scanpaths, im_w, im_h, visual_angle=24):
    if len(gt_scanpaths) == 0:
        return {}
    task_names = np.unique([traj['task'] for traj in gt_scanpaths])
    all_fixs = []
    prior_maps = {}
    for task in task_names:
        Xs = np.concatenate([
            traj['X'][1:] for traj in gt_scanpaths
            if traj['split'] == 'train' and traj['task'] == task
        ])
        Ys = np.concatenate([
            traj['Y'][1:] for traj in gt_scanpaths
            if traj['split'] == 'train' and traj['task'] == task
        ])
        fixs = np.stack([Xs, Ys]).T.astype(np.int32)
        prior_maps[task] = convert_fixations_to_map(fixs,
                                                    im_w,
                                                    im_h,
                                                    smooth=True,
                                                    visual_angle=visual_angle)
        all_fixs.append(fixs)
    all_fixs = np.concatenate(all_fixs)
    prior_maps['all'] = convert_fixations_to_map(all_fixs,
                                                 im_w,
                                                 im_h,
                                                 smooth=True,
                                                 visual_angle=visual_angle)
    return prior_maps


def get_IoM(bb1, bb2):
    """
    Calculate the Intersection over min area of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # # compute the intersection over union by taking the intersection
    # # area and dividing it by the sum of prediction + ground-truth
    # # areas - the interesection area
    # iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    iom = intersection_area / float(min(bb1_area, bb2_area))
    iom = np.clip(iom, 0, 1)
    # assert iom >= 0.0, "out-of-range IoM = {:.3f}".format(iom)
    # assert iom <= 1.0, "out-of-range IoM = {:.3f}".format(iom)
    
    return iom

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(
        mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

    
def transform_fixations(normalized_fixations, is_padding, pa, sample_scanpath=True, return_highres=False):
    """Transform a batch of fixation into sequences of categorical labels."""
    def transform(normalized_fixations, is_padding, patch_num, sample_scanpath):
        fixs = (normalized_fixations * torch.Tensor(patch_num)).to(torch.long)
        labels = patch_num[0] * fixs[:, :, 1] + fixs[:, :, 0]
        labels += 1 + int(sample_scanpath)
        labels[is_padding==1] = pa.pad_idx
        if sample_scanpath:
            term_idx = is_padding.argmax(dim=1)
            labels[torch.arange(len(labels))[term_idx>0], term_idx[term_idx>0]]= pa.eos_idx
        return labels.to(torch.long)
    fix_seq = transform(normalized_fixations, is_padding, [pa.im_w//32, pa.im_h//32], sample_scanpath)
    if return_highres:
        fix_seq_high = transform(normalized_fixations, is_padding, [pa.im_w//4, pa.im_h//4], sample_scanpath)
        return fix_seq, fix_seq_high
    else:
        return fix_seq, None
    

def create_mask(tgt, pad_idx, device):
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return tgt_mask, tgt_padding_mask
