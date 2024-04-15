import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from scipy.stats import norm
from scipy.interpolate import griddata

def img_cortical_magnif_tsr(imgtsr, pnt, grid_func, demo=True):
    if imgtsr.ndim == 4:
        imgtsr.squeeze_(0)
    _, H, W = imgtsr.shape
    XX_intp, YY_intp = grid_func(imgtsr, pnt)
    grid = torch.stack([(torch.tensor(XX_intp) / W * 2 - 1),  # normalize the value to -1, 1
                        (torch.tensor(YY_intp) / H * 2 - 1)],  # normalize the value to -1, 1
                       dim=2).unsqueeze(0).float()
    # print(grid.shape) # 1, H, W, 2
    img_cm = F.grid_sample(imgtsr.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros')
    img_cm.squeeze_(0)
    if demo:
        # % Visualize the Manified plot.
        pY, pX = pnt
        figh, axs = plt.subplots(3, 1, figsize=(6, 12))
        axs[0].imshow(img_cm.permute([1,2,0]))
        axs[0].axis("off")
        axs[1].imshow(imgtsr.permute([1,2,0]))
        axs[1].axis("off")
        axs[1].scatter([pX], [pY], c='r', s=16, alpha=0.5)
        axs[2].scatter(XX_intp[::2, ::2].flatten(), YY_intp[::2, ::2].flatten(), c="r", s=0.25, alpha=0.2)
        axs[2].set_xlim([0, W])
        axs[2].set_ylim([0, H])
        axs[2].invert_yaxis()
        plt.show()
    return img_cm


def cortical_magnif_tsr_demo(imgtsr, pnt, grid_func, subN=2):
    """ Demo the cortical magnification transform. 
    Inputs: 
        imgtsr: Image tensor with shape (C, H, W) or (1, C, H, W)
        pnt: (X, Y) tuple. 
        grid_func: a function with signature 
            `XX_intp, YY_intp = grid_func(imgtsr, pnt)`
            It generates interpolation grid of X and Y. 

    Parameters:
        subN: subsample the sampling grid by `subN` to show in the figure.
    """
    if imgtsr.ndim == 4:
        imgtsr.squeeze_(0)
    _, H, W = imgtsr.shape
    XX_intp, YY_intp = grid_func(imgtsr, pnt)
    grid = torch.stack([(torch.tensor(XX_intp) / W * 2 - 1),  # normalize the value to -1, 1
                        (torch.tensor(YY_intp) / H * 2 - 1)],  # normalize the value to -1, 1
                       dim=2).unsqueeze(0).float()
    img_cm = F.grid_sample(imgtsr.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros')
    img_cm.squeeze_(0)
    # % Visualize the Manified plot.
    pY, pX = pnt
    figh, axs = plt.subplots(3, 1, figsize=(6, 12))
    axs[0].imshow(img_cm.permute([1,2,0]))
    axs[0].axis("off")
    axs[1].imshow(imgtsr.permute([1,2,0]))
    axs[1].axis("off")
    axs[1].scatter([pX], [pY], c='r', s=16, alpha=0.5)
    axs[2].scatter(XX_intp[::subN, ::subN].flatten(),
                   YY_intp[::subN, ::subN].flatten(), c="r", s=0.25, alpha=0.2)
    axs[2].set_xlim([0, W])
    axs[2].set_ylim([0, H])
    axs[2].invert_yaxis()
    plt.show()
    return figh, img_cm, imgtsr


def linear_separable_gridfun(imgtsr, pnt, ):
    _, H, W = imgtsr.shape
    Hhalf, Whalf = H // 2, W // 2
    Hsum = Hhalf * (Hhalf + 1) / 2
    Wsum = Whalf * (Whalf + 1) / 2
    pY, pX = pnt
    UpDelta = pY / Hsum
    LeftDelta = pX / Wsum
    DownDelta = (H - pY) / Hsum
    RightDelta = (W - pX) / Wsum
    Left_ticks = np.cumsum(LeftDelta * np.arange(Whalf, 0, -1))
    Right_ticks = np.cumsum(RightDelta * np.arange(1, Whalf + 1, 1)) + pX
    Up_ticks = np.cumsum(UpDelta * np.arange(Hhalf, 0, -1))
    Down_ticks = np.cumsum(DownDelta * np.arange(1, Hhalf + 1, 1)) + pY
    X_ticks = np.hstack((Left_ticks, Right_ticks))
    Y_ticks = np.hstack((Up_ticks, Down_ticks))
    XX_intp, YY_intp = np.meshgrid(X_ticks, Y_ticks, )
    return XX_intp, YY_intp


def normal_gridfun(imgtsr, pnt, cutoff_std=2.25):
    """

    cutoff_std: where to cut off the normal distribution. too large will make the sampling at center
        too dense!
    """
    _, H, W = imgtsr.shape
    Hhalf, Whalf = H // 2, W // 2
    Hdensity = norm.pdf(np.linspace(0, cutoff_std, Hhalf))
    Wdensity = norm.pdf(np.linspace(0, cutoff_std, Whalf))
    H_delta = (1 / Hdensity)
    W_delta = (1 / Wdensity)
    Hsum = H_delta.sum()
    Wsum = W_delta.sum()
    pY, pX = pnt
    UpDelta = pY / Hsum
    LeftDelta = pX / Wsum
    DownDelta = (H - pY) / Hsum
    RightDelta = (W - pX) / Wsum
    Left_ticks = np.cumsum(LeftDelta * W_delta[::-1])
    Right_ticks = np.cumsum(RightDelta * W_delta[::]) + pX
    Up_ticks = np.cumsum(UpDelta * H_delta[::-1])
    Down_ticks = np.cumsum(DownDelta * H_delta[::]) + pY
    X_ticks = np.hstack((Left_ticks, Right_ticks))
    Y_ticks = np.hstack((Up_ticks, Down_ticks))
    XX_intp, YY_intp = np.meshgrid(X_ticks, Y_ticks, )
    return XX_intp, YY_intp


def radial_quad_isotrop_gridfun(imgtsr, pnt, fov=20, K=20, cover_ratio=None):
    _, H, W = imgtsr.shape
    Hhalf, Whalf = H // 2, W // 2
    pY, pX = pnt
    maxdist = np.sqrt(max(H - pY, pY)**2 + max(W - pX, pX)**2)  # in pixel
    grid_y, grid_x = np.mgrid[-Hhalf+0.5:Hhalf+0.5, -Whalf+0.5:Whalf+0.5]
    ecc2 = grid_y**2 + grid_x**2 # R2
    ecc = np.sqrt(ecc2)
    # RadDistTfm = lambda R, R2 : (R < fov) * R + (R > fov) * (R**2 - fov**2 + fov)
    RadDistTfm = lambda R: (R < fov) * R + \
        (R > fov) * ((R + K) ** 2 / 2 / (fov + K) + fov - (fov + K) / 2)

    ecc_tfm = RadDistTfm(ecc, )
    coef = maxdist / ecc_tfm.max()
    if cover_ratio is not None:
        if type(cover_ratio) in [list, tuple]:
            ratio = np.random.uniform(cover_ratio[0], cover_ratio[1])
            coef = coef * np.sqrt(ratio)
        else:
            coef = coef * np.sqrt(cover_ratio)  # may not be optimal
    XX_intp = pX + coef * ecc_tfm * (grid_x / ecc)  # cosine
    YY_intp = pY + coef * ecc_tfm * (grid_y / ecc)  # sine
    return XX_intp, YY_intp


def radial_exp_isotrop_gridfun(imgtsr, pnt, slope_C=2.0, cover_ratio=None):
    _, H, W = imgtsr.shape
    Hhalf, Whalf = H // 2, W // 2
    pY, pX = pnt
    maxdist = np.sqrt(max(H - pY, pY)**2 + max(W - pX, pX)**2)  # in pixel
    grid_y, grid_x = np.mgrid[-Hhalf+0.5:Hhalf+0.5, -Whalf+0.5:Whalf+0.5]
    ecc2 = grid_y**2 + grid_x**2  # R2
    ecc = np.sqrt(ecc2)
    if type(slope_C) in [list, tuple]:
        slope = np.random.uniform(slope_C[0], slope_C[1])
    else:
        slope = slope_C  # may not be optimal
    RadDistTfm = lambda R: 1 / slope * (np.exp(slope * R / np.sqrt(Hhalf**2 + Whalf**2)) - 1)  # normalization
    ecc_tfm = RadDistTfm(ecc, )
    coef = maxdist / ecc_tfm.max()
    if cover_ratio is not None:
        if type(cover_ratio) in [list, tuple]:
            ratio = np.random.uniform(cover_ratio[0], cover_ratio[1])
            coef = coef * np.sqrt(ratio)
        else:
            coef = coef * np.sqrt(cover_ratio)  # may not be optimal
    XX_intp = pX + coef * ecc_tfm * (grid_x / ecc)  # cosine
    YY_intp = pY + coef * ecc_tfm * (grid_y / ecc)  # sine
    return XX_intp, YY_intp
#%%
if __name__ == "__main__":
    #%%
    from scipy.misc import face
    from skimage.transform import rescale
    img = rescale(face(), (0.25, 0.25, 1))
    imgtsr = torch.tensor(img).permute([2,0,1]).float()
    #%%
    img_cm = img_cortical_magnif_tsr(imgtsr, (80, 120), linear_separable_gridfun, demo=True)
    #%%
    img_cm = img_cortical_magnif_tsr(imgtsr, (80, 120), normal_gridfun, demo=True)
    #%%
    img_cm = img_cortical_magnif_tsr(imgtsr, (10, 190),
                                     lambda img,pnt: radial_quad_isotrop_gridfun(img, pnt, fov=20, K=20), demo=True)
    #%%
    img_cm = img_cortical_magnif_tsr(imgtsr, (100, 30),
            lambda img, pnt: radial_exp_isotrop_gridfun(img, pnt, slope_C=2.0, cover_ratio=0.4), demo=True)

    #%%  linear_separable
    rndMagnif = get_RandomMagnifTfm(grid_generator="radial_quad_isotrop", bdr=16, fov=20, K=0, cover_ratio=(0.05, 1))
    mtg = make_grid([rndMagnif(imgtsr) for _ in range(9)], nrow=3)
    mtg_pil = ToPILImage()(mtg)
    mtg_pil.show()
    #%%
    rndMagnif = get_RandomMagnifTfm(grid_generator="normal", bdr=16, fov=30, K=5)
    mtg = make_grid([rndMagnif(imgtsr) for _ in range(9)], nrow=3)
    mtg_pil = ToPILImage()(mtg)
    mtg_pil.show()
    #%%
    rndMagnif = get_RandomMagnifTfm(grid_generator="radial_exp_isotrop", bdr=64, slope_C=(0.75, 3.0), cover_ratio=(0.1, 0.5))
    mtg = make_grid([rndMagnif(imgtsr) for _ in range(9)], nrow=3)
    mtg_pil = ToPILImage()(mtg)
    mtg_pil.show()
