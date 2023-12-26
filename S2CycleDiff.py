from torch.nn import init
import torch.nn.functional as F
from einops import rearrange, repeat
# from tqdm.notebook import tqdm
from functools import partial
import math, os, copy
from DataLoad import *
from tqdm import tqdm
from prettytable import PrettyTable
import scipy.io as sio
import imgvision as iv

"""
    Define U-net Architecture:
    Approximate reverse diffusion process by using U-net
    U-net of SR3 : U-net backbone + Positional Encoding of time + Multihead Self-Attention
"""

import torch
import torch.nn as nn

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def calculate_sam(target_data, reference_data):
    # 归一化目标数据和参考数据
    b, c, h, w = target_data.shape
    target_data = target_data.reshape(b, c, h*w).permute(0,2,1)
    reference_data = reference_data.reshape(b, c, h * w).permute(0, 2, 1)
    target_data_norm = torch.nn.functional.normalize(target_data, dim=2)
    reference_data_norm = torch.nn.functional.normalize(reference_data, dim=2)

    # 计算点积
    dot_product = torch.einsum('bnc,bnc->bn', target_data_norm, reference_data_norm)

    # 计算长度乘积
    length_product = torch.norm(target_data_norm, dim=2) * torch.norm(reference_data_norm, dim=2)

    # 计算SAM光谱角
    sam = torch.acos(dot_product / length_product)
    sam_mean = torch.mean(torch.mean(sam, dim=1))
    return sam_mean


def extract(a, t, x_shape):
    """
    从给定的张量a中检索特定的元素。t是一个包含要检索的索引的张量，
    这些索引对应于a张量中的元素。这个函数的输出是一个张量，
    包含了t张量中每个索引对应的a张量中的元素
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# Linear Multi-head Self-attention
class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(SelfAtt, self).__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)
        self.att = att_num

    def forward(self, x):
        x_org = x
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

        return x_org+self.att*self.proj(out)


class Cross_Att(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(Cross_Att, self).__init__()
        self.att = att_num
        self.groupnorm_1 = nn.GroupNorm(norm_groups, channel_dim)
        self.groupnorm_2 = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv_1 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.qkv_2 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)

        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

        self.downsample = nn.Sequential(nn.Conv2d(channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 2, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),
                                        )

        self.upsample = nn.Sequential(nn.Conv2d(2 * channel_dim, 1 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      )
    def forward(self, x, y, mode):
        b, c, h, w = x.size()
        x_org = x
        if mode == 'spe':
            b, c, h, w = x.size()
            x = self.groupnorm_1(x)
            y = self.groupnorm_1(y)
            qkv_1 = rearrange(self.qkv_1(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(self.qkv_2(y), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]
            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)
        else:
            x = self.groupnorm_2(x)
            y = self.groupnorm_2(y)
            if h == 512:
                times = h/64
            else:
                times = h/20
            n = np.log(times)/np.log(2)
            for i in range(int(n)):
                x = self.downsample[2 * i](x)
                x = self.downsample[2 * i+1](x)

            for i in range(int(n)):
                y = self.downsample[2 * i](y)
                y = self.downsample[2 * i+1](y)

            b, c, h, w = x.size()

            x = x.reshape(b, c, h * w).repeat(1, 1, 3)
            y = y.reshape(b, c, h * w).repeat(1, 1, 3)
            qkv_1 = rearrange(x, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(y, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]

            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads (h w) c -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

            for i in range(int(n)):
                l = int(n)-1-i
                out = self.upsample[2 * l](out)
                out = self.upsample[2 * l+1](out)

        return x_org+self.att*self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=False):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        x = y + self.res_conv(x)
        return x


class ResBlock_skip(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=True):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        y = self.block1(x)

        return y+self.res_conv(x)


class SGPD(nn.Module):
    def __init__(self, in_channel=37, out_channel=34, skip_input=31, inner_channel=64, norm_groups=32,
                 channel_mults=[1, 2, 4, 8, 8], res_blocks=3, dropout=0, img_size=128):
        super().__init__()

        self_att = []
        cros_att = []
        dim_out = [inner_channel, inner_channel * 2, inner_channel * 2]
        for i in reversed(range(len(dim_out))):
            self_att.append(SelfAtt(dim_out[i], num_heads=1, norm_groups=norm_groups))

        self.self_att = nn.ModuleList(self_att)

        for j in reversed(range(len(dim_out))):
            cros_att.append(Cross_Att(dim_out[j], num_heads=1, norm_groups=norm_groups))

        self.cros_att = nn.ModuleList(cros_att)

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        pre_channel_skip = inner_channel
        feat_channels = [pre_channel]
        feat_channels_skips = [pre_channel]

        now_res = img_size

        # Downsampling stage of SGPD
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Skip stage of SGPD
        skip_downs = [nn.Conv2d(skip_input, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                skip_downs.append(ResBlock_skip(
                    pre_channel_skip, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, att=False))
                pre_channel_skip = channel_mult
            if not is_last:
                feat_channels_skips.append(channel_mult)
                skip_downs.append(Downsample(pre_channel_skip))
                now_res = now_res // 2
        self.skip_downs = nn.ModuleList(skip_downs)

        # Upsampling stage of SGPD
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]

            for i in range(0, res_blocks + 1):
                ups.append(ResBlock(
                    pre_channel + feat_channels.pop()*2, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult

            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, skip_input, noise_level, mode = None):
        # Embedding of time step with noise coefficient alpha
        t = self.noise_level_mlp(noise_level)

        feats_skip = []
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        k = 0
        for i, layer in enumerate(self.skip_downs):

            # skip_input =
            skip_input = layer(skip_input)
            feats_skip.append(skip_input)

        for layer in self.mid:
            x = layer(x, t)
        z = 0
        for i, layer in enumerate(self.ups):

            if isinstance(layer, ResBlock):
                if i == 0:
                    x = layer(torch.cat([x, feats.pop(), feats_skip.pop()], dim=1), t)
                elif isinstance(self.ups[i - 1], Upsample):
                    temp_feats_skip = feats_skip.pop()
                    temp_feats = feats.pop()
                    x = layer(torch.cat([x, self.self_att[z](temp_feats_skip), self.cros_att[z](temp_feats, temp_feats_skip, mode=mode)], dim=1), t)
                    z = z+1
                else:
                    x = layer(torch.cat([x, feats.pop(), feats_skip.pop()], dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


"""
    Define Diffusion process framework to train desired model:
    Forward Diffusion process:
        Given original image x_0, apply Gaussian noise ε_t for each time step t
        After proper length of time step, image x_T reachs to pure Gaussian noise
    Objective of model f :
        model f is trained to predict actual added noise ε_t for each time step t
"""

class Diffusion(nn.Module):
    def __init__(self, model_spe, model_spa, device, img_size, LR_size, channels=3):
        super().__init__()
        self.channels = channels
        self.model_spe = model_spe.to(device)
        self.model_spa = model_spa.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device

        self.PSF = nn.Sequential(transforms.GaussianBlur((3, 3), sigma=1),
                                 nn.Conv2d(1, 1, 4, 4, 0, bias=False)).to(device)

        self.PSF_31 = nn.Conv2d(31, 31, 4, 4, 0).to(device)
        self.PSF_31.weight = torch.nn.Parameter(self.PSF[1].weight.repeat(31, 31, 1, 1))

        self.PSF_3 = nn.Conv2d(3, 3, 4, 4, 0, bias=False).to(device)
        self.PSF_3.weight = torch.nn.Parameter(self.PSF[1].weight.repeat(3, 3, 1, 1))

        self.SRF = nn.Conv2d(31, 3, kernel_size=1, stride=1, padding=0, bias=False).to(device)
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

        # complementary fusion block
        self.fuse = nn.Sequential(
            nn.Conv2d(31*2, 31*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(31*2, 31, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(31, 31, kernel_size=3, stride=1, padding=1),
        ).to(device)

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac = 0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal,
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior


    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, condition_skip=None):
        batch_size, c = x.shape[0], condition_x.shape[1]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        # x_recon = self.predict_start(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))

        if c < 5:
            x_start = self.model_spe(torch.cat([condition_x, x], dim=1), skip_input=self.upsample(condition_skip), noise_level = noise_level, mode='spe')
        else:
            condition_x = self.upsample(condition_x)
            x_start = self.model_spa(torch.cat([condition_x, x], dim=1), skip_input=condition_skip, noise_level = noise_level, mode='spa')

        posterior_mean = (
                self.posterior_mean_coef1[t] * x_start.clamp(-1, 1) +
                self.posterior_mean_coef2[t] * x
        )

        posterior_variance = self.posterior_log_variance_clipped[t]

        mean, posterior_log_variance = posterior_mean, posterior_variance
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, img_spa, img_spe, t, clip_denoised=True, condition_lrHS=None, condition_hrMS=None):

        mean1, log_variance1 = self.p_mean_variance(x=img_spa, t=t, clip_denoised=clip_denoised, condition_x=condition_lrHS, condition_skip=condition_hrMS)

        mean2, log_variance2 = self.p_mean_variance(x=img_spe, t=t, clip_denoised=clip_denoised, condition_x=condition_hrMS, condition_skip=condition_lrHS)

        noise = torch.randn_like(img_spa) if t > 0 else torch.zeros_like(img_spa)
        return mean1 + noise * (0.5 * log_variance1).exp(), mean2 + noise * (0.5 * log_variance2).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, lrHS, hrMS, gtHS):
        img_spa = torch.rand_like(gtHS, device=gtHS.device)
        img_spe = torch.rand_like(gtHS, device=gtHS.device)
        for i in reversed(range(0, self.num_timesteps)):

            img_spa, img_spe = self.p_sample(img_spa, img_spe, i, condition_lrHS=lrHS, condition_hrMS=hrMS)

        img = self.fuse(torch.cat([img_spe, img_spa], dim=1))
        return img, img_spa, img_spe

    # Compute loss to train the model
    def p_losses(self, x_in):
        x_start = x_in
        lr_imgs = transforms.Resize(self.img_size)(transforms.Resize(self.LR_size)(x_in))
        b, c, h, w = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t], size=b)).to(x_start.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)

        noise = torch.randn_like(x_start).to(x_start.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * x_start + (1 - sqrt_alpha ** 2).sqrt() * noise
        # The model predict actual noise added at time step t
        pred_noise = self.model(torch.cat([lr_imgs, x_noisy], dim=1), noise_level=sqrt_alpha)

        return self.loss_func(noise, pred_noise), pred_noise, lr_imgs

    def net(self, gtHS, hrMS, lrHS):

        gtHS = gtHS
        lrMS = hrMS
        lrHS = lrHS

        b, c, h, w = gtHS.shape
        t = torch.randint(1, schedule_opt['n_timestep'], size=(b,))
        sqrt_alpha_cumprod_t = extract(torch.from_numpy(self.sqrt_alphas_cumprod_prev), t, gtHS.shape)
        sqrt_alpha = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1).type(torch.float32).to(gtHS.device)
        noise = torch.randn_like(gtHS).to(gtHS.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * gtHS + (1 - sqrt_alpha ** 2).sqrt() * noise
        # The bilateral model predict actual x0 added at time step t
        pred_x0_spe = self.model_spe(torch.cat([lrMS, x_noisy], dim=1), skip_input = self.upsample(lrHS), noise_level=sqrt_alpha, mode='spe')
        pred_x0_spa = self.model_spa(torch.cat([self.upsample(lrHS), x_noisy], dim=1), skip_input = lrMS, noise_level=sqrt_alpha, mode='spa')

        # complementary fusion
        pred_x0_fuse = self.fuse(torch.cat([pred_x0_spe.detach(), pred_x0_spa.detach()], dim=1))

        pred_HSI = self.PSF_31(pred_x0_spe)
        pred_MSI = self.SRF(pred_x0_spa)

        loss_sam_fuse = 0

        loss_1 = self.loss_func(gtHS, pred_x0_spe) / int(b * c * h * w)
        loss_2 = self.loss_func(gtHS, pred_x0_spa)/ int(b*c*h*w)
        loss_3 = self.loss_func(pred_x0_spa, pred_x0_spe)/int(b*c*h*w)
        loss_4 = self.loss_func(lrHS, pred_HSI)/int(b*c*(h/4)*(w/4))
        loss_5 = self.loss_func(hrMS, pred_MSI)/int(b*3*h*w)
        loss_6 = self.loss_func(self.PSF_3(hrMS), self.SRF(lrHS))/int(b*3*(h/4)*(w/4))
        loss_7 = self.loss_func(gtHS, pred_x0_fuse)/int(b*c*h*w)

        return loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_sam_fuse

    def forward(self, gtHS, hrMS, lrHS, *args, **kwargs):
        return self.net(gtHS, hrMS, lrHS,  *args, **kwargs)


# Class to train & test desired model
class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader,
                 schedule_opt, save_path, load_path=None, load=True,
                 in_channel=62, out_channel=31, inner_channel=64, norm_groups=8,
                 channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-3, distributed=False):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.LR_size = LR_size

        # Bilateral Spe-Spa Guided Pyramid Denoing model
        model_spe = SGPD(34, 31, 31, inner_channel, norm_groups, channel_mults, res_blocks, dropout, img_size)
        model_spa = SGPD(62, 31, 3, inner_channel, norm_groups, channel_mults, res_blocks, dropout, img_size)

        self.sr3 = Diffusion(model_spe, model_spa, device, img_size, LR_size, out_channel)
        # Apply weight initialization & set loss & set noise schedule
        self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)

            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):

        train = True

        for i in range(epoch):
            i = i
            train_loss = 0
            self.sr3.train()
            randn1 = np.random.randint(0, 100)
            loss_1_epoch = 0
            loss_2_epoch = 0
            loss_3_epoch = 0
            loss_4_epoch = 0
            loss_5_epoch = 0
            loss_6_epoch = 0
            loss_7_epoch = 0
            loss_sam_epoch = 0

            if train:
                for step, [gtHS, hrMS, lrHS] in enumerate(tqdm(self.dataloader)):
                    # 高光谱和全色图像
                    gtHS = gtHS.to(self.device).type(torch.float32)
                    hrMS = hrMS.to(self.device).type(torch.float32)
                    lrHS = lrHS.to(self.device).type(torch.float32)
                    b, c, h, w = gtHS.shape
                    randn2 = np.random.randint(0, b)
                    self.optimizer.zero_grad()
                    loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_sam = self.sr3(gtHS, hrMS, lrHS)

                    loss_1 = loss_1.sum()
                    loss_2 = loss_2.sum()
                    loss_3 = loss_3.sum()
                    loss_4 = loss_4.sum()
                    loss_5 = loss_5.sum()
                    loss_6 = loss_6.sum()
                    loss_7 = loss_7.sum()
                    loss_sam = loss_sam

                    if i<=0:
                        loss = loss_1 + loss_2 + loss_4 + loss_5 +0.1*loss_6
                    else:
                        loss = loss_1 + loss_2 + 0.1*loss_4 + 0.1*loss_5 +0.1*loss_6+ loss_7
                    loss.backward()
                    self.optimizer.step()

                    loss_1_epoch = loss_1.item() + loss_1_epoch
                    loss_2_epoch = loss_2.item() + loss_2_epoch
                    loss_3_epoch = loss_3.item() + loss_3_epoch
                    loss_4_epoch = loss_4.item() + loss_4_epoch
                    loss_5_epoch = loss_5.item() + loss_5_epoch
                    loss_6_epoch = loss_6.item() + loss_6_epoch
                    loss_7_epoch = loss_7.item() + loss_7_epoch
                    loss_sam_epoch = loss_sam_epoch

                    train_loss += loss.item()

                print('epoch: {}'.format(i))
                print('损失函数:')
                x = PrettyTable()
                x.add_column("loss", ['value'])
                x.add_column("loss_all", [train_loss / float(len(self.dataloader))])
                x.add_column("loss_1", [loss_1_epoch / float(len(self.dataloader))])
                x.add_column("loss_2", [loss_2_epoch / float(len(self.dataloader))])
                x.add_column("loss_3", [loss_3_epoch / float(len(self.dataloader))])
                x.add_column("loss_4", [loss_4_epoch / float(len(self.dataloader))])
                x.add_column("loss_5", [loss_5_epoch / float(len(self.dataloader))])
                x.add_column("loss_6", [loss_6_epoch / float(len(self.dataloader))])
                x.add_column("loss_7", [loss_7_epoch / float(len(self.dataloader))])
                x.add_column("loss_sam_spe", [loss_sam_epoch / float(len(self.dataloader))])
                print(x)

            if (i + 1) % verbose == 0:
                self.sr3.eval()
                test_data = copy.deepcopy(next(iter(self.testloader)))
                [gtHS, hrMS, lrHS] = test_data
                gtHS = gtHS.to(self.device).type(torch.float32)
                hrMS = hrMS.to(self.device).type(torch.float32)
                lrHS = lrHS.to(self.device).type(torch.float32)

                b, c, h, w = gtHS.shape

                randn3 = np.random.randint(0, b)
                gtHS = gtHS[randn3]
                hrMS = hrMS[randn3]
                lrHS = lrHS[randn3]
                # Transform to low-resolution images
                # Save example of test images to check training
                plt.figure(figsize=(15, 10))
                plt.subplot(2, 3, 1)
                plt.axis("off")
                plt.title("MSI")
                plt.imshow(np.transpose(torchvision.utils.make_grid(hrMS,
                                                                    nrow=2, padding=1, normalize=True).cpu(),
                                        (1, 2, 0)))

                plt.subplot(2, 3, 2)
                plt.axis("off")
                plt.title("HSI_input")
                # A = self.test(test_img, test_lrHS_img)
                plt.imshow(np.transpose(torchvision.utils.make_grid(lrHS.cpu(),
                                                                    nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :,[5, 15, 30]])

                plt.subplot(2, 3, 3)
                plt.axis("off")
                plt.title("spa")
                self.save(self.save_path, i)
                A, spa, spe= self.test(lrHS.unsqueeze(0), hrMS.unsqueeze(0), gtHS.unsqueeze(0))
                plt.imshow(np.transpose(torchvision.utils.make_grid(spa.cpu(),
                                                                    nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :,[5, 15, 30]])

                plt.subplot(2, 3, 4)
                plt.axis("off")
                plt.title("spe")
                plt.imshow(np.transpose(torchvision.utils.make_grid(spe.cpu(),
                                                                    nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :,
                           [5, 15, 30]])

                plt.subplot(2, 3, 5)
                plt.axis("off")
                plt.title("fuse_output")
                plt.imshow(np.transpose(torchvision.utils.make_grid(A.cpu(),
                                                                    nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :,
                           [5, 15, 30]])

                plt.subplot(2, 3, 6)
                plt.axis("off")
                plt.title("GT")
                plt.imshow(np.transpose(torchvision.utils.make_grid(gtHS.cpu(),
                                                                    nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :,
                           [5, 15, 30]])
                sio.savemat('result/cave_{}.mat'.format(i), {'output': A.cpu().numpy()})
                sio.savemat('result/cave_spa_{}.mat'.format(i), {'output': spa.cpu().numpy()})
                sio.savemat('result/cave_spe_{}.mat'.format(i), {'output': spe.cpu().numpy()})
                plt.savefig('./img/cave_x0_fuse/SuperResolution_Result_test' + str(i) + '.jpg')
                plt.show()
                plt.close()

                # Save model weight

                Metric = iv.spectra_metric(gtHS.permute(1, 2, 0).cpu().detach().numpy(), A[0].permute(1, 2, 0).cpu().detach().numpy(), 4)
                PSNR = Metric.PSNR()
                SAM = Metric.SAM()
                SSIM = Metric.SSIM()
                MSE = Metric.MSE()
                ERGAS = Metric.ERGAS()
                print('评价指标:')
                y = PrettyTable()
                y.add_column("Index", ['value'])
                y.add_column("PSNR", [PSNR])
                y.add_column("SAM", [SAM])
                y.add_column("SSIM", [SSIM])
                y.add_column("MSE", [MSE])
                y.add_column("ERGAS", [ERGAS])
                print(y)

    def test(self, lrHS, hrMS, gtHS):
        lrHS = lrHS
        hrMS = hrMS
        gtHS = gtHS
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(lrHS, hrMS, gtHS)
            else:
                result_SR = self.sr3.super_resolution(lrHS, hrMS, gtHS)
        self.sr3.train()
        return result_SR

    def save(self, save_path, i):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path+'SR3_model_epoch-{}.pt'.format(i))

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")


if __name__ == "__main__":
    batch_size = 12
    LR_size = 32
    img_size = 128
    root = './defusion/data/cave/'

    train_data = DataLoad(root, 'train')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data = DataLoad(root, 'test')
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    schedule_opt = {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-4, 'linear_end': 0.002}

    sr3 = SR3(device, img_size=img_size, LR_size=LR_size, loss_type='l1',
              dataloader=train_dataloader, testloader=test_dataloader, schedule_opt=schedule_opt,
              save_path='./cycle/model/cave_x0_fuse/',
              load_path='./model/cave_x0_fuse/SR3_model_epoch-799.pt', load=False,
              inner_channel=64,
              norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0, res_blocks=2, lr=1e-4, distributed=True)
    sr3.train(epoch=10000, verbose=100)



