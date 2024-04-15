import torch
from .utils import foveal2mask, real_foveation_batch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DCB_Env:
    """
    Environment for low- and high-res DCB under 
    inverse reinforcement learning
    """

    def __init__(self,
                 pa,
                 max_step,
                 mask_size,
                 status_update_mtd,
                 device,
                 ret_inhibition=True,
                 init_mtd='center'):
        self.pa = pa
        self.init = init_mtd
        self.max_step = max_step + 1  # one more step to hold the initial step
        self.inhibit_return = ret_inhibition
        self.mask_size = mask_size
        self.status_update_mtd = status_update_mtd
        self.device = device

    def observe(self, accumulate=True):
        active_indices = self.is_active
        if torch.sum(active_indices) == 0:
            return None, None

        if self.step_id > 0:
            # update state with high-res feature
            remap_ratio = self.pa.patch_num[0] / float(self.states.size(-1))
            lastest_fixation_on_feats = self.fixations[:, self.step_id].to(
                dtype=torch.float32) / remap_ratio
            px = lastest_fixation_on_feats[:, 0]
            py = lastest_fixation_on_feats[:, 1]
            masks = []
            for i in range(self.batch_size):
                mask = foveal2mask(px[i].item(), py[i].item(),
                                   self.pa.fovea_radius, self.states.size(-2),
                                   self.states.size(-1))
                mask = torch.from_numpy(mask).to(self.device)
                mask = mask.unsqueeze(0).repeat(self.states.size(1), 1, 1)
                masks.append(mask)
            masks = torch.stack(masks)

            if accumulate:
                self.states = (1 - masks) * self.states + masks * self.hr_feats
            else:
                self.states = (1 -
                               masks) * self.lr_feats + masks * self.hr_feats
            self.history_map = (1 -
                                masks[:, 0]) * self.history_map + masks[:, 0]
        ext_states = self.states.clone()

        return ext_states, self.fix_ind_map.clone()

    def get_reward(self, prob_old, prob_new):
        return torch.zeros(self.batch_size, device=self.device)

    def step(self, act_batch):
        self.step_id += 1
        assert self.step_id < self.max_step, "Error: Exceeding maximum step!"

        # update fixation
        py, px = act_batch // self.pa.patch_num[
            0], act_batch % self.pa.patch_num[0]
        self.fixations[:, self.step_id, 1] = py
        self.fixations[:, self.step_id, 0] = px
        self.fix_ind_map[torch.arange(self.batch_size), self.step_id, py, px] = True

        # update action mask
        if self.inhibit_return:
            action_idx = torch.arange(0,
                                      self.batch_size,
                                      device=self.device,
                                      dtype=torch.long)
            if self.mask_size == 0:
                self.action_mask[action_idx, act_batch] = 1
            else:
                bs = self.action_mask.size(0)
                px, py = px.to(dtype=torch.long), py.to(dtype=torch.long)
                self.action_mask = self.action_mask.view(
                    bs, self.pa.patch_num[1], -1)
                for i in range(bs):
                    self.action_mask[i,
                                     max(py[i] - self.mask_size, 0):py[i] +
                                     self.mask_size + 1,
                                     max(px[i] - self.mask_size, 0):px[i] +
                                     self.mask_size + 1] = 1

                self.action_mask = self.action_mask.view(bs, -1)

        obs = self.observe()
        self.status_update(act_batch)

        return obs, self.status.clone()

    def status_update(self, act_batch):
        if self.status_update_mtd == 'SOT':  # stop on target
            done = self.label_coding[torch.arange(self.batch_size
                                                  ), 0, act_batch]
        elif self.status_update_mtd == 'SGTL':  # stop at ground-truth length
            done = self.gt_length == self.step_id - 1
        elif self.status_update_mtd == 'SASA':  # stop at stopping action
            done = act_batch == self.pa.patch_count
        else:
            raise NotImplementedError

        done[self.status > 0] = 2
        self.status = done.to(torch.uint8)

    def step_back(self):
        self.fixations[:, self.step_id] = 0
        self.step_id -= 1

    def reset(self):
        self.step_id = 0  # step id of the environment
        self.fixations = torch.zeros((self.batch_size, self.max_step, 2),
                                     dtype=torch.long,
                                     device=self.device)
        self.status = torch.zeros(self.batch_size,
                                  dtype=torch.uint8,
                                  device=self.device)
        self.is_active = torch.ones(self.batch_size,
                                    dtype=torch.uint8,
                                    device=self.device)
        self.states = self.lr_feats.clone()

        if self.inhibit_return:
            self.action_mask = self.init_action_mask.clone()

        self.history_map = self.init_history_map.clone()

        self.fix_ind_map = self.init_fix_ind_map.clone()

        # random initialization
        if self.init == 'random':
            raise NotImplementedError
        # center initialization
        elif self.init == 'center':
            self.fixations[:, 0] = torch.tensor(
                [[self.pa.patch_num[0] / 2, self.pa.patch_num[1] / 2]],
                dtype=torch.long,
                device=self.device)
            if self.inhibit_return:
                bs = self.action_mask.size(0)
                self.action_mask = self.action_mask.view(bs, self.pa.patch_num[1],
                                                        -1)
                px, py = int(self.pa.patch_num[0] / 2), int(self.pa.patch_num[1] /
                                                            2)
                self.action_mask[:, py - self.mask_size:py + self.mask_size +
                                1, px - self.mask_size:px + self.mask_size +
                                1] = 1
                self.action_mask = self.action_mask.view(bs, -1)
        elif self.init == 'manual':
            self.fixations[:, 0, 0] = self.init_fix[:, 0]
            self.fixations[:, 0, 1] = self.init_fix[:, 1]
        else:
            raise NotImplementedError

    def set_data(self, data):
        self.label_coding = data['label_coding'].to(self.device)
        self.img_names = data['img_name']
        self.cat_names = data['cat_name']
        self.init_fix = data['init_fix'].to(self.device)
        self.init_action_mask = data['action_mask'].to(self.device)
        self.init_history_map = data['history_map'].to(self.device)
        self.init_fix_ind_map = data['fix_ind_map'].to(self.device)
        self.task_ids = data['task_id'].to(self.device)
        self.lr_feats = data['lr_feats'].to(self.device)
        self.hr_feats = data['hr_feats'].to(self.device)
        self.batch_size = self.hr_feats.size(0)
        if self.inhibit_return:
            self.action_mask = data['action_mask'].to(self.device).view(
                self.batch_size, -1)
        else:
            # self.action_mask = torch.zeros(self.batch_size,
            #                                self.pa.patch_count,
            #                                dtype=torch.uint8)
            self.action_mask = None
        self.reset()
    def clone(self):
        return copy.deepcopy(self)

class CFI_Env:
    """
    Environment for Foveal Vision approach under inverse reinforcement learning
    """

    def __init__(self,
                 pa,
                 state_enc,
                 max_step,
                 mask_size,
                 status_update_mtd,
                 device,
                 ret_inhibition=True,
                 init_mtd='center'):
        self.pa = pa
        self.init = init_mtd
        self.max_step = max_step + 1  # one more step to hold the initial step
        self.ret_inhibition = ret_inhibition
        self.state_enc = state_enc
        self.mask_size = mask_size
        self.device = device
        self.status_update_mtd = status_update_mtd

    def observe(self, accumulate=True):
        active_indices = self.is_active
        if torch.sum(active_indices) == 0:
            return None

        if accumulate:
            fixs = self.fixations[:, :self.step_id + 1].to(torch.float)
        else:
            fixs = self.fixations[:, self.step_id:self.step_id + 1].to(
                torch.float)

        x, _ = real_foveation_batch(self.pyramid, fixs, self.pa)

        if self.state_enc:
            with torch.no_grad():
                x = self.state_enc(x)

        return x

    def get_reward(self, prob_old, prob_new):
        return torch.zeros(self.batch_size, device=self.device)

    def step(self, act_batch):
        self.step_id += 1
        assert self.step_id < self.max_step, "Error: Exceeding maximum step!"

        # update fixation
        py, px = act_batch // self.pa.patch_num[
            0], act_batch % self.pa.patch_num[0]
        self.fixations[:, self.step_id, 1] = py
        self.fixations[:, self.step_id, 0] = px

        # update action mask
        before_action_mask = self.action_mask.clone()
        if self.ret_inhibition:
            action_idx = torch.arange(0,
                                      self.batch_size,
                                      device=self.device,
                                      dtype=torch.long)
            if self.mask_size == 0:
                self.action_mask[action_idx, act_batch] = 1
            else:
                bs = self.action_mask.size(0)
                px, py = px.to(dtype=torch.long), py.to(dtype=torch.long)
                self.action_mask = self.action_mask.view(
                    bs, self.pa.patch_num[1], -1)
                for i in range(bs):
                    self.action_mask[i,
                                     max(py[i] - self.mask_size, 0):py[i] +
                                     self.mask_size + 1,
                                     max(px[i] - self.mask_size, 0):px[i] +
                                     self.mask_size + 1] = 1

                self.action_mask = self.action_mask.view(bs, -1)
        # if self.action_mask.sum() - before_action_mask.sum() == 0:
        #     print('error!!!')
        #     action_mask = before_action_mask.view(bs, self.pa.patch_num[1], -1)
        #     print(
        #         act_batch, py, px,
        #         action_mask[0,
        #                     max(py[i] -
        #                         self.mask_size, 0):py[i] + self.mask_size + 1,
        #                     max(px[i] - self.mask_size, 0):px[i] +
        #                     self.mask_size + 1])

        obs = self.observe()
        self.status_update(act_batch)

        return obs, self.status

    def status_update(self, act_batch):
        if self.status_update_mtd == 'SOT':  # stop on target
            done = self.label_coding[torch.arange(self.batch_size
                                                  ), 0, act_batch]
        elif self.status_update_mtd == 'SGTL':  # stop at ground-truth length
            done = self.gt_length == self.step_id - 1
        elif self.status_update_mtd == 'SASA':  # stop at stopping action
            done = act_batch == self.pa.patch_count
        else:
            raise NotImplementedError

        done[self.status > 0] = 2
        self.status = done.to(torch.uint8)

    def step_back(self):
        self.fixations[:, self.step_id] = 0
        self.step_id -= 1

    def reset(self):
        self.step_id = 0  # step id of the environment
        self.fixations = torch.zeros((self.batch_size, self.max_step, 2),
                                     dtype=torch.long,
                                     device=self.device)

        self.status = torch.zeros(self.batch_size,
                                  dtype=torch.uint8,
                                  device=self.device)
        self.is_active = torch.ones(self.batch_size,
                                    dtype=torch.uint8,
                                    device=self.device)
        self.action_mask = self.init_action_mask.clone()

        # random initialization
        if self.init == 'random':
            raise NotImplementedError
        # center initialization
        elif self.init == 'center':
            self.fixations[:, 0] = torch.tensor(
                [[self.pa.patch_num[0] / 2, self.pa.patch_num[1] / 2]],
                dtype=torch.long,
                device=self.device)
            bs = self.action_mask.size(0)
            self.action_mask = self.action_mask.view(bs, self.pa.patch_num[1],
                                                     -1)
            px, py = int(self.pa.patch_num[0] / 2), int(self.pa.patch_num[1] /
                                                        2)
            self.action_mask[:, py - self.mask_size:py + self.mask_size +
                             1, px - self.mask_size:px + self.mask_size +
                             1] = 1

            self.action_mask = self.action_mask.view(bs, -1)
        elif self.init == 'manual':
            self.fixations[:, 0, 0] = self.init_fix[:, 0]
            self.fixations[:, 0, 1] = self.init_fix[:, 1]
        else:
            raise NotImplementedError

    def set_data(self, data):
        self.pyramid = data['pyramid'].to(self.device)
        self.batch_size = self.pyramid.size(0)
        self.label_coding = data['label_coding'].to(self.device)
        self.init_action_mask = data['action_mask'].to(self.device)
        self.img_names = data['img_name']
        self.task_ids = data['task_id'].to(self.device)
        # self.gt_length = data['gt_length'].to(
        #    self.device)  # ground-truth scanpath legnth
        self.reset()



class FFN_Env:
    """
    Environment for foveal feature pyramid approach under inverse reinforcement learning.
    """

    def __init__(self,
                 pa,
                 max_step,
                 mask_size,
                 status_update_mtd,
                 device,
                 ret_inhibition=True,
                 accumulate_state=True,
                 init_mtd='center'):
        self.pa = pa
        self.init = init_mtd
        self.max_step = max_step + 1  # one more step to hold the initial step
        self.ret_inhibition = ret_inhibition
        self.mask_size = mask_size
        self.device = device
        self.status_update_mtd = status_update_mtd
        self.accumulate = accumulate_state

    def observe(self):
        active_indices = self.is_active
        if torch.sum(active_indices) == 0:
            return None, None

        num_fixs_left = self.max_step - self.step_id - 1
        last_fixs = self.fixations[:, self.step_id].unsqueeze(1).expand(
            -1, num_fixs_left, -1)
        made_fixations = torch.cat(
            [self.fixations[:, :self.step_id + 1], last_fixs],
            axis=1).to(dtype=torch.float32)
        made_fixations /= torch.tensor(self.pa.patch_num).to(self.device)
        return self.im_tensors, made_fixations

    def step(self, act_batch):
        self.step_id += 1
        assert self.step_id < self.max_step, "Error: Exceeding maximum step!"

        # update fixation
        py, px = act_batch // self.pa.patch_num[
            0], act_batch % self.pa.patch_num[0]
        self.fixations[:, self.step_id, 1] = py
        self.fixations[:, self.step_id, 0] = px

        # update action mask
        before_action_mask = self.action_mask.clone()
        if self.ret_inhibition:
            action_idx = torch.arange(0,
                                      self.batch_size,
                                      device=self.device,
                                      dtype=torch.long)
            if self.mask_size == 0:
                self.action_mask[action_idx, act_batch] = 1
            else:
                bs = self.action_mask.size(0)
                px, py = px.to(dtype=torch.long), py.to(dtype=torch.long)
                self.action_mask = self.action_mask.view(
                    bs, self.pa.patch_num[1], -1)
                for i in range(bs):
                    self.action_mask[i,
                                     max(py[i] - self.mask_size, 0):py[i] +
                                     self.mask_size + 1,
                                     max(px[i] - self.mask_size, 0):px[i] +
                                     self.mask_size + 1] = 1

                self.action_mask = self.action_mask.view(bs, -1)

        obs = self.observe()
        self.status_update(act_batch)

        return obs, self.status

    def status_update(self, act_batch):
        if self.status_update_mtd == 'SOT':  # stop on target
            done = self.label_coding[torch.arange(self.batch_size
                                                  ), 0, act_batch]
        elif self.status_update_mtd == 'SGTL':  # stop at ground-truth length
            done = self.gt_length == self.step_id - 1
        elif self.status_update_mtd == 'SASA':  # stop at stopping action
            done = act_batch == self.pa.patch_count
        else:
            raise NotImplementedError

        done[self.status > 0] = 2
        self.status = done.to(torch.uint8)

    def step_back(self):
        self.fixations[:, self.step_id] = 0
        self.step_id -= 1

    def reset(self):
        self.step_id = 0  # step id of the environment
        self.fixations = torch.zeros((self.batch_size, self.max_step, 2),
                                     dtype=torch.long,
                                     device=self.device)

        self.status = torch.zeros(self.batch_size,
                                  dtype=torch.uint8,
                                  device=self.device)
        self.is_active = torch.ones(self.batch_size,
                                    dtype=torch.uint8,
                                    device=self.device)
        self.action_mask = self.init_action_mask.clone()

        # random initialization
        if self.init == 'random':
            raise NotImplementedError
        # center initialization
        elif self.init == 'center':
            self.fixations[:, 0] = torch.tensor(
                [[int(self.pa.patch_num[0] / 2), int(self.pa.patch_num[1] / 2)]],
                dtype=torch.long,
                device=self.device)
            bs = self.action_mask.size(0)
            self.action_mask = self.action_mask.view(bs, self.pa.patch_num[1],
                                                     -1)
            px, py = int(self.pa.patch_num[0] / 2), int(self.pa.patch_num[1] /
                                                        2)
            self.action_mask[:, py - self.mask_size:py + self.mask_size +
                             1, px - self.mask_size:px + self.mask_size +
                             1] = 1

            self.action_mask = self.action_mask.view(bs, -1)
        elif self.init == 'manual':
            self.fixations[:, 0, 0] = self.init_fix[:, 0]
            self.fixations[:, 0, 1] = self.init_fix[:, 1]
        else:
            raise NotImplementedError

    def set_data(self, data):
        self.im_tensors = data['im_tensor'].to(self.device)
        self.batch_size = self.im_tensors.size(0)
        self.label_coding = data['label_coding'].to(self.device)
        self.init_action_mask = data['action_mask'].to(self.device)
        self.img_names = data['img_name']
        self.task_ids = data['task_id'].to(self.device)
        self.reset()
