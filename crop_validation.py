import torch
from IPython import embed
import pdb
import numpy as np
# import pyflow

def lr_crop_index(n, N, D, base_size, overlap):
    n_end = D if n == N - 1 else (n + 1) * base_size + overlap
    n_start = n_end - base_size - overlap
    return n_start, n_end


def hr_crop_index(n, N, D, Dmod, base_size, overlap):
    if n == 0:
        n_start = 0
        n_end = (n + 1) * base_size + (overlap // 2 if N > 1 else overlap)
        pn_start = n_start
        pn_end = n_end
    elif n == N - 1:
        n_end = D
        pn_end = base_size + overlap
        if Dmod > overlap:
            n_start = n_end - Dmod
            pn_start = pn_end - Dmod
        else:
            n_start = n_end - base_size - overlap // 2
            pn_start = overlap // 2
    else:
        n_start = n * base_size + overlap // 2
        n_end = n_start + base_size
        pn_start = overlap // 2
        pn_end = pn_start + base_size
    return n_start, n_end, pn_start, pn_end


def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,-1,1)
    return flow


def forward_crop(x, model, lq_size=64, scale=1, overlap=16, flow_opt=False):

    # Assert the required dimension.
    assert lq_size == 64 or 48, "Default patch size of LR images during training and validation should be {}.".format(lq_size)
    assert overlap == 16 or 12, "Default overlap of patches during validation should be {}.".format(overlap)

    # Prepare for the image crops.
    # print(x.shape)
    base_size = lq_size - overlap
    B, T, C, H, W = x.shape
    # print(x.shape)

    I = H // base_size
    Hmod = H % base_size
    if Hmod > overlap:
        I += 1

    J = W // base_size
    Wmod = W % base_size
    if Wmod > overlap:
        J += 1

    # print(I, Hmod, J, Wmod)

    # Crop the entire image into 64 x 64 patches. Concatenate the crops along the batch dimension.
    x_crops = []
    for i in range(I):
        i_start, i_end = lr_crop_index(i, I, H, base_size, overlap)
        for j in range(J):
            j_start, j_end = lr_crop_index(j, J, W, base_size, overlap)
            x_crop = x[:, :, :, i_start: i_end, j_start: j_end]
            if flow_opt:
                x_crop, _ = model(x_crop,B,T,phase='test')
            else:
                x_crop = model(x_crop,B,T,phase='test')
            x_crop = x_crop.reshape(B,T,x_crop.shape[1],x_crop.shape[2],x_crop.shape[3])
            x_crops.append(x_crop)
            
    x_crops = torch.cat(x_crops, dim=0)
    #print(x_crops.shape)
    # Execute the model
    # if flow_opt:
    #     x_crops, _ = model(x_crops)
    # else:
    #     x_crops = model(x_crops)

    if len(x_crops.shape) == 5:
        x_crops = x_crops[:, T//2, :, :, :]

    # Calculate the enlarged dimension.
    H, W = H * scale, W * scale
    Hmod, Wmod = Hmod * scale, Wmod * scale
    base_size, overlap = base_size * scale, overlap * scale
    # print(H, W, Hmod, Wmod, base_size, overlap)
    # print('Second')
    # Convert the SR crops to an entire image
    if len(x_crops.shape) == 4:
        x = torch.zeros(B, C, H, W)
        for i in range(I):
            i_start, i_end, pi_start, pi_end = hr_crop_index(i, I, H, Hmod, base_size, overlap)
            for j in range(J):
                j_start, j_end, pj_start, pj_end = hr_crop_index(j, J, W, Wmod, base_size, overlap)
                # print(i_start, i_end, j_start, j_end)
                # print(pi_start, pi_end, pj_start, pj_end)
                B_start = B * (i * J + j)
                B_end = B_start + B
                # print(B_start, B_end)
                x[:, :, i_start: i_end, j_start: j_end] \
                    = x_crops[B_start: B_end, :, pi_start: pi_end, pj_start: pj_end]
    # elif len(x_crops.shape) == 5:
    #     x = torch.zeros(B, T, C, H, W)
    #     for t in range(T):
    #         for i in range(I):
    #             i_start, i_end, pi_start, pi_end = hr_crop_index(i, I, H, Hmod, base_size, overlap)
    #             for j in range(J):
    #                 j_start, j_end, pj_start, pj_end = hr_crop_index(j, J, W, Wmod, base_size, overlap)
    #                 B_start = B * (i * J + j)
    #                 B_end = B_start + B
    #                 x[:, t, :, i_start: i_end, j_start: j_end] \
    #                     = x_crops[B_start: B_end, t, :, pi_start: pi_end, pj_start: pj_end]
    return x


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output
