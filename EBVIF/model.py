import torch
import torch.nn as nn
import math
import numpy
import utils
import random

def transform(data, msfa_size=4, spatial_ratio=2):
    # rotation
    if numpy.random.uniform() < 0.5:
        rn = random.randint(1, 3)
        data = torch.rot90(data, rn, [2, 3])
 
    # flip
    if numpy.random.uniform() < 0.5:
        r = numpy.random.uniform()
        if r < 0.25:
            data = torch.flip(data, [2])
        elif r < 0.5:
            data = torch.flip(data, [3])
        elif r < 0.75:
            data = torch.flip(data, [2])
            data = torch.flip(data, [3])
        else:
            data = torch.flip(data, [3])
            data = torch.flip(data, [2])

    # shift
    if numpy.random.uniform() < 0.5:
        i = random.randint(0, msfa_size*spatial_ratio-1)
        j = random.randint(0, msfa_size*spatial_ratio-1)
        data = torch.roll(data, (i, j), (2, 3))

    return data

class CNNs_v1(nn.Module):
    def __init__(self, Cin, Cout):
        super(CNNs_v1, self).__init__()
        self.conv1 = nn.Conv2d(Cin, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, Cout*2, 3, 1, 1)

        self.relu = nn.ReLU()
    
    def forward(self, mosaic, pan):
        mosaic_reshape = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=4)
        mosaic_interpolate = torch.nn.functional.interpolate(mosaic_reshape, scale_factor=8, mode="bicubic")
        x = torch.cat((mosaic_interpolate, pan), 1)
        y = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))
        c = y.shape[1]
        mu_log, sigma_log = y[:, :c//2], y[:, c//2:]
        return mu_log, sigma_log

class CNNs_v2(nn.Module):
    def __init__(self, Cin, Cout):
        super(CNNs_v2, self).__init__()
        self.conv1 = nn.Conv2d(Cin, 64, 3, 2, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, Cout*2, 3, 2, 1)

        self.relu = nn.ReLU()
    
    def forward(self, mosaic, pan):
        mosaic_reshape = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=4)
        mosaic_interpolate = torch.nn.functional.interpolate(mosaic_reshape, scale_factor=8, mode="bicubic")
        x = torch.cat((mosaic_interpolate, pan), 1)
        y = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))
        c = y.shape[1]
        mu_log, sigma_log = y[:, :c//2], y[:, c//2:]
        return mu_log, sigma_log

class Degrade_SRF(nn.Module):
    def __init__(self, Cin, Cout):
        super(Degrade_SRF, self).__init__()
        self.spec_res = nn.Conv2d(Cin, Cout, 1, 1, 0, bias=False)
        self.spec_res.weight.data = torch.ones_like(self.spec_res.weight)
        self.spec_res.weight.data /= self.spec_res.weight.data.sum()
    
    def forward(self, x):
        y = self.spec_res(x)
        return y
    
class Degrade_BDM(nn.Module):
    def __init__(self, ksize=13):
        super(Degrade_BDM, self).__init__()

        self.psf = nn.Conv2d(1, 1, ksize, 1, ksize//2, bias=False)
        self.psf.weight.data = torch.ones_like(self.psf.weight)
        self.psf.weight.data /= self.psf.weight.data.sum()
    
    def forward(self, x, msfa_kernel):
        z = []
        for i in range(x.shape[1]):
            z.append(self.psf(x[:, i:i+1, :, :]))
        z = torch.cat(z, 1)
        z = torch.nn.functional.conv2d(z, msfa_kernel, bias=None, stride=msfa_kernel.shape[2], groups=z.shape[1])
        return z

class FuseNet(nn.Module):
    def __init__(self, args):
        super(FuseNet, self).__init__()
        self.mu_sigma_ny = CNNs_v1(args.num_bands+1, 1)
        self.mu_sigma_nz = CNNs_v2(args.num_bands+1, args.num_bands)
        self.mu_sigma_f_and_g = CNNs_v1(args.num_bands+1, args.num_bands)
        
        self.degrade_srf = Degrade_SRF(args.num_bands, 1)
        self.degrade_bdm = Degrade_BDM(ksize=13)
        
    def forward_for_inference(self, z, y, msfa_kernel):
        # estimate ny
        mu_ny_log, sigma_ny_log = self.mu_sigma_ny(z, y)
        mu_ny = torch.exp(mu_ny_log)
        sigma_ny = torch.exp(sigma_ny_log / 2)
        est_ny = mu_ny + torch.randn(mu_ny.shape).to(mu_ny.device) * sigma_ny

        # estimate nz
        mu_nz_log, sigma_nz_log = self.mu_sigma_nz(z, y)
        mu_nz = torch.exp(mu_nz_log)
        sigma_nz = torch.exp(sigma_nz_log / 2)
        est_nz = mu_nz + torch.randn(mu_nz.shape).to(mu_nz.device) * sigma_nz
        est_nz = torch.nn.functional.pixel_shuffle(est_nz, 4)

        # estimate f
        denoise_y = y - est_ny
        denoise_z = z - est_nz
        mu_f_log, sigma_f_log = self.mu_sigma_f_and_g(denoise_z, denoise_y)
        mu_f = torch.exp(mu_f_log)
        sigma_f = torch.exp(sigma_f_log/ 2)
        rand_val1 = torch.randn(mu_f.shape).to(mu_f.device)
        est_f = mu_f + rand_val1 * sigma_f

        # estimate g
        degrade_srf = self.degrade_srf(est_f)
        sparsity_y = denoise_y - degrade_srf
        degrade_bdm = self.degrade_bdm(est_f, msfa_kernel)
        degrade_bdm = torch.nn.functional.pixel_shuffle(degrade_bdm, 4)
        sparsity_z = denoise_z - degrade_bdm
        mu_g_log, sigma_g_log = self.mu_sigma_f_and_g(sparsity_z, sparsity_y)
        mu_g = torch.exp(mu_g_log)
        sigma_g = torch.exp(sigma_g_log / 2)
        rand_val2 = torch.randn(mu_g.shape).to(mu_g.device)
        est_g = mu_g + rand_val2 * sigma_g

        return est_f, est_g, est_ny, est_nz
    
    def forward_for_train(self, z, y, msfa_kernel):
        # estimate ny
        mu_ny_log, sigma_ny_log = self.mu_sigma_ny(z, y)
        mu_ny = torch.exp(mu_ny_log)
        sigma_ny = torch.exp(sigma_ny_log / 2)
        est_ny = mu_ny + torch.randn(mu_ny.shape).to(mu_ny.device) * sigma_ny

        # estimate nz
        mu_nz_log, sigma_nz_log = self.mu_sigma_nz(z, y)
        mu_nz = torch.exp(mu_nz_log)
        sigma_nz = torch.exp(sigma_nz_log / 2)
        est_nz = mu_nz + torch.randn(mu_nz.shape).to(mu_nz.device) * sigma_nz
        est_nz = torch.nn.functional.pixel_shuffle(est_nz, 4)

        # estimate f
        denoise_y = y - est_ny
        denoise_z = z - est_nz
        mu_f_log, sigma_f_log = self.mu_sigma_f_and_g(denoise_z, denoise_y)
        mu_f = torch.exp(mu_f_log)
        sigma_f = torch.exp(sigma_f_log/ 2)
        rand_val1 = torch.randn(mu_f.shape).to(mu_f.device)
        est_f = mu_f + rand_val1 * sigma_f

        # estimate g
        degrade_srf = self.degrade_srf(est_f)
        sparsity_y = denoise_y - degrade_srf
        degrade_bdm = self.degrade_bdm(est_f, msfa_kernel)
        degrade_bdm = torch.nn.functional.pixel_shuffle(degrade_bdm, 4)
        sparsity_z = denoise_z - degrade_bdm
        mu_g_log, sigma_g_log = self.mu_sigma_f_and_g(sparsity_z, sparsity_y)
        mu_g = torch.exp(mu_g_log)
        sigma_g = torch.exp(sigma_g_log / 2)
        rand_val2 = torch.randn(mu_g.shape).to(mu_g.device)
        est_g = mu_g + rand_val2 * sigma_g

        # degrade 
        degrade_srf = self.degrade_srf(est_f+est_g) + est_ny
        degrade_bdm = torch.nn.functional.pixel_shuffle(self.degrade_bdm(est_f+est_g, msfa_kernel), 4) + est_nz

        # loss w. r. t. y, z
        loss_y = 0.5 * ((y - degrade_srf)**2).mean()
        loss_z = 0.5 * ((z - degrade_bdm)**2).mean()
        
        loss = loss_y + loss_z

        # loss w.r.t my
        sigma_my_zero = 1e-3
        loss_my = 0.5 * (
                    torch.log(sigma_ny) - math.log(sigma_my_zero) \
                    +(sigma_my_zero**2 + mu_ny**2) / sigma_ny**2
                    ).mean()
        
        # loss w.r.t mz
        sigma_mz_zero = 1e-3
        loss_mz = 0.5 * (
                    torch.log(sigma_nz) - math.log(sigma_mz_zero) \
                    +(sigma_mz_zero**2 + mu_nz**2) / sigma_nz**2
                    ).mean()

        loss += loss_my + loss_mz

        # EI
        fused = transform(est_f+est_g, msfa_size=4, spatial_ratio=2).detach()
        degrade_y1 = self.degrade_srf(fused)
        degrade_z1 = self.degrade_bdm(fused, msfa_kernel)
        degrade_z1 = torch.nn.functional.pixel_shuffle(degrade_z1, 4)

        # estimate f'
        denoise_y = degrade_y1
        denoise_z = degrade_z1
        mu_f_log, sigma_f_log = self.mu_sigma_f_and_g(denoise_z, denoise_y)
        mu_f = torch.exp(mu_f_log)
        sigma_f = torch.exp(sigma_f_log/ 2)
        est_f = mu_f + rand_val1 * sigma_f

        # estimate g'
        degrade_srf = self.degrade_srf(est_f)
        sparsity_y = denoise_y - degrade_srf
        degrade_bdm = self.degrade_bdm(est_f, msfa_kernel)
        degrade_bdm = torch.nn.functional.pixel_shuffle(degrade_bdm, 4)
        sparsity_z = denoise_z - degrade_bdm
        mu_g_log, sigma_g_log = self.mu_sigma_f_and_g(sparsity_z, sparsity_y)
        mu_g = torch.exp(mu_g_log)
        sigma_g = torch.exp(sigma_g_log / 2)
        est_g = mu_g + rand_val2 * sigma_g

        loss += 0.5 * ((fused - est_f - est_g)**2).mean()

        return loss

    def forward(self, z, y, msfa_kernel, inference_flag=False):
        if inference_flag == True:
            return self.forward_for_inference(z, y, msfa_kernel)
        else:
            return self.forward_for_train(z, y, msfa_kernel)