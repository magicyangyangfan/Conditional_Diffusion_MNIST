''' 
This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from Dataset import Dataset
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_res=False):
        super().__init__()
        self.is_res = is_res
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.GELU()
        if is_res and in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        if self.is_res:
            res = x if self.res_conv is None else self.res_conv(x)
            out = (res + out) / 1.414
        return out


class UnetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, is_res=True)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        return self.pool(self.block(x))


class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block1 = ResidualConvBlock(out_ch*2, out_ch, is_res=True)
        self.block2 = ResidualConvBlock(out_ch, out_ch, is_res=True)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.block2(self.block1(x))



class EmbedFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.fc(x)


class ContextUnet(nn.Module):
    def __init__(self, in_ch, base_ch=128, cond_dim=10):
        super().__init__()
        # Encoder
        self.conv0 = ResidualConvBlock(in_ch, base_ch, is_res=True)   # 128x128
        self.down1 = UnetDown(base_ch, base_ch*2)                     # 64x64
        self.down2 = UnetDown(base_ch*2, base_ch*4)                   # 32x32
        # Embeddings
        self.to_vec = nn.AdaptiveAvgPool2d(1)
        self.temb1 = EmbedFC(1, base_ch*4)
        self.temb2 = EmbedFC(1, base_ch*2)
        self.cemb1 = EmbedFC(cond_dim, base_ch*4)
        self.cemb2 = EmbedFC(cond_dim, base_ch*2)
        # Decoder (two up steps)
        self.up1 = UnetUp(base_ch*4, base_ch*2)  # 32->64
        self.up2 = UnetUp(base_ch*2, base_ch)    # 64->128
        self.out = nn.Conv2d(base_ch*2, in_ch, 1)

    def forward(self, x, c_flat, t_norm, mask):
        B = x.size(0)
        # Encode
        x0 = self.conv0(x)       # [B, base_ch, H, W]
        d1 = self.down1(x0)      # [B, base_ch*2, H/2, W/2]
        d2 = self.down2(d1)      # [B, base_ch*4, H/4, W/4]
        # Temporal embedding
        temb1 = self.temb1(t_norm).view(B, -1, 1, 1)
        temb2 = self.temb2(t_norm).view(B, -1, 1, 1)
        # Context embedding with mask
        keep = mask.view(B,1)
        c_flat = c_flat * keep
        cemb1 = self.cemb1(c_flat).view(B, -1, 1, 1)
        cemb2 = self.cemb2(c_flat).view(B, -1, 1, 1)
        # Decode
        u1 = self.up1(d2 * cemb1 + temb1, d1)  # [B, base_ch*2, H/2, W/2]
        u2 = self.up2(u1 * cemb2 + temb2, x0)  # [B, base_ch,   H,   W]
        # Final conv
        return self.out(torch.cat([u2, x0], dim=1))



def ddpm_schedules(beta1, beta2, T):
    betas = torch.linspace(beta1, beta2, T)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


class DDPM(nn.Module):
    def __init__(self, model, T=1000, betas=(1e-4,0.02), device='cuda'):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.T = T
        betas, alphas, alphas_bar = ddpm_schedules(betas[0], betas[1], T)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.loss_fn = nn.MSELoss()

    def forward(self, x0, c_flat):
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        a_bar = self.alphas_bar[t].view(B,1,1,1)
        xt = torch.sqrt(a_bar)*x0 + torch.sqrt(1-a_bar)*noise
        # context mask
        mask = (torch.rand(B, device=self.device) > 0.1).float()
        t_norm = (t.float()/self.T).unsqueeze(1)
        eps_pred = self.model(xt, c_flat, t_norm, mask)
        return self.loss_fn(noise, eps_pred)

    @torch.no_grad()
    def sample(self, n, size, c_flat, guide_w=0.0):
        x = torch.randn(n, *size, device=self.device)
        B = n
        # mask: first half unconditional, second half conditional
        mask = torch.cat([torch.zeros(B), torch.ones(B)], dim=0).to(self.device)
        c = c_flat.repeat(2,1)
        x_seq = []
        x_seq.append(x)
        for i in reversed(range(self.T)):
            t_norm = (torch.full((2*B,), i, device=self.device).float()/self.T).unsqueeze(1)
            x_in = x.repeat(2,1,1,1)
            eps = self.model(x_in, c, t_norm, mask)
            eps1, eps2 = eps.chunk(2, dim=0)
            eps = (1+guide_w)*eps2 - guide_w*eps1
            a = self.alphas[i]
            a_bar = self.alphas_bar[i]
            z = torch.randn_like(x) if i>0 else 0
            x = (1/torch.sqrt(a))*(x - (1-a)/torch.sqrt(1-a_bar)*eps) + torch.sqrt(self.betas[i])*z
            if i % 50 == 0:
                x_seq.append(x)
        return x, x_seq


def train_mnist():

    # hardcoding these here
    n_epoch = 1000
    batch_size =16
    n_T = 500 # 500
    device = "cuda:0"
    csv_file = '/home/yangyang/Projects/Conditional_Diffusion_MNIST/data/AlSiMgCu.txt'
    n_classes = 7
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = './data/diffusion_outputs3/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    unet = ContextUnet(in_ch=1, base_ch=128, cond_dim=n_classes)
    ddpm = DDPM(unet,  n_T, (1e-4, 0.02), device=device)
    ddpm.to(device)

    # optionally load a model
    ddpm.load_state_dict(torch.load("data/diffusion_outputs2/model_99.pth"))


    dataset = Dataset(csv_file=csv_file,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5)),
                            ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.view(c.size(0), -1).float().to(device)  # [B, n_classes]
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        if ep % 100 == 0 or ep == int(n_epoch-1) or ep ==0:
            ddpm.eval()
            with torch.no_grad():
                n_sample = n_classes
                for w_i, w in enumerate(ws_test):
                    # Create a dataloader with batch_size matching n_sample
                    dataloader_sample = DataLoader(dataset, batch_size=n_sample, shuffle=True, num_workers=5)
                    
                    # Get the first batch and extract c_i
                    for _, c_sample, img_name in dataloader_sample:
                        c_i = c_sample.to(device)
                        c_i = c_i.view(c_i.size(0), -1).float()
                        break
                    x_gen,_ = ddpm.sample(n_sample, (1, 128, 128), c_i, guide_w=w)

                    # append some real images at bottom, order by class also
                    # x_real = torch.Tensor(x_gen.shape).to(device)
                    # for k in range(n_classes):
                    #     for j in range(int(n_sample/n_classes)):
                    #         try: 
                    #             idx = torch.squeeze((c == k).nonzero())[j]
                    #         except:
                    #             idx = 0
                    #         x_real[k+(j*n_classes)] = x[idx]

                    # x_all = torch.cat([x_gen, x_real])
                    with open(save_dir + "generated_images.txt", "a") as f:
                        for name in img_name:
                            f.write(name + "\n")
                    grid = make_grid(x_gen, nrow=10)
                    save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                    print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                    # if ep%5==0 or ep == int(n_epoch-1):
                    #     # create gif of images evolving over time, based on x_gen_store
                    #     fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    #     def animate_diff(i, x_gen_store):
                    #         print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                    #         plots = []
                    #         for row in range(int(n_sample/n_classes)):
                    #             for col in range(n_classes):
                    #                 axs[row, col].clear()
                    #                 axs[row, col].set_xticks([])
                    #                 axs[row, col].set_yticks([])
                    #                 # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                    #                 plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                    #         return plots
                        # ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                        # ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                        # print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
            # optionally save model
            # if save_model and ep == int(n_epoch-1):
                torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
                print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()

