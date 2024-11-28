import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime



import utils
from utils import compute_fid_and_save_samples
from parameter import get_paras
from model.Discriminator import Discriminator
from model.Generator import Generator


def train(args):
    k,n_critic, batch_size, lr, num_iterations, device, betas, checkpoint_interval, num_workers,OUTPUT_DIR, DATA_DIR = \
        args.k, args.n_critic, args.batch_size, args.lr, args.num_iterations, args.device, \
        args.betas, args.checkpoint_interval, args.num_workers, args.output_dir, args.data_dir
    dataloader = utils.load_data(batch_size,num_workers)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    G = Generator(judge=False).to(device)
    D = Discriminator(judge=False).to(device)

    fid = 0
    print(f"""
    初始化训练...
    训练参数:
    ├── 模型参数:
    │   ├── k: {k}
    │   └── betas: {betas}
    ├── 训练配置:
    │   ├── n_critic: {n_critic}
    │   ├── batch_size: {batch_size}
    │   ├── learning_rate: {lr}
    │   ├── num_iterations: {num_iterations}
    │   ├── checkpoint_interval: {checkpoint_interval}
    │   └── num_workers: {num_workers}
    ├── 硬件配置:
    │   └── device: {device}
    └── 路径配置:
        ├── output_dir: {OUTPUT_DIR}
        └── data_dir: {DATA_DIR}
    """)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(OUTPUT_DIR, f'结果_{timestamp}')
    samples_dir = os.path.join(save_dir, '生成样本')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    print(f"结果将保存到: {save_dir}")

    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas)

    # 创建真实图像目录用于FID计算
    real_temp_dir = '/image/real'
    os.makedirs(real_temp_dir, exist_ok=True)

    print("\n保存真实图像用于FID计算...")
    num_r = 0
    for i, (real_imgs, _) in enumerate(dataloader):
        for j, img in enumerate(real_imgs):
            vutils.save_image(
                img,
                f"{real_temp_dir}/{i}_{j}.png",
                normalize=True,
                value_range=(-1, 1)
            )
            if num_r>=5000:
                break
        if num_r >= 5000:  # 保存足够的图像后停止
            break

    G_losses = []
    D_losses = []
    FIDs = []
    grad_flows = []

    print("\n开始训练循环...")
    for iteration in tqdm(range(num_iterations)):
        
        # 训练判别器
        for _ in range(n_critic):
            optimizerD.zero_grad()
            try:
                real_data = next(iter(dataloader))[0].to(device)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                real_data = next(dataloader_iterator)[0].to(device)
            batch_size = real_data.size(0)
            d_loss_totle = 0
            
            
            for i in range(batch_size):
                real_sample = real_data[i:i + 1]
                z = torch.randn(1, 128).to(device)
                fake_sample = G(z)
                
                mu = torch.rand(1, 1, 1, 1).to(device)
                x_interpolated = (1 - mu) * real_sample  + mu * fake_sample
                x_interpolated.requires_grad_(True)

                d_real = D(real_sample)
                d_fake = D(fake_sample.detach())
                wasserstein_distance = torch.mean(d_real) - torch.mean(d_fake)


                d_interpolated = D(x_interpolated)
                gradients = torch.autograd.grad(
                    outputs=d_interpolated,
                    inputs=x_interpolated,
                    grad_outputs=torch.ones_like(d_interpolated).to(device),
                    create_graph=True,
                    retain_graph=True,
                )[0]

                # 计算梯度惩罚项
                gradients = gradients.view(batch_size, -1)
                gradient_norm = gradients.norm(2, dim=1)
                gradient_penalty = k * ((gradient_norm.mean()-1)**2)

                # 总判别器损失
                d_loss_single = -wasserstein_distance + gradient_penalty
                d_loss_totle += d_loss_single
                d_loss = d_loss_totle/batch_size

            d_loss.backward()
            optimizerD.step()
            D_losses.append(d_loss.item())
            grad_flow_d = torch.cat([p.grad.abs().mean().unsqueeze(0)
                                     for p in D.parameters() if p.grad is not None])


        # 步骤6：更新生成器
        optimizerG.zero_grad()
        z = torch.randn(batch_size, 128).to(device)
        fake_data = G(z)
        g_loss = -torch.mean(D(fake_data))

        g_loss.backward()
        optimizerG.step()

        G_losses.append(g_loss.item())

        # 记录生成器梯度流
        grad_flow = torch.cat([p.grad.abs().mean().unsqueeze(0)
                               for p in G.parameters() if p.grad is not None])
        grad_flows.append(grad_flow.mean().item())

        # 检查点：保存模型和计算FID
        if iteration % checkpoint_interval == 0:
            print(f"\n当前迭代: {iteration}/{num_iterations}")
            print(f"判别器损失: {d_loss.item():.4f}, 生成器损失: {g_loss.item():.4f}")

            G.eval()
            fid = compute_fid_and_save_samples(
                G=G,
                device=device,
                real_temp_dir=real_temp_dir,
                samples_dir=samples_dir,
                iteration=iteration,
                batch_size=batch_size
            )
            FIDs.append(fid)
            G.train()

            # 保存检查点
            utils.save_checkpoint(
                save_dir=save_dir,
                iteration=iteration,
                G=G,
                D=D,
                optimizerG=optimizerG,
                optimizerD=optimizerD,
                G_losses=G_losses,
                D_losses=D_losses,
                FIDs=FIDs,
                grad_flows=grad_flows
            )


if __name__ == "__main__":
    args = get_paras("wgan-gp")
    utils.set_seed()
    print("数据集加载完成")
    device = utils.choice_device()

    print(f"模型已初始化并移至{device}")
    train(args)

    print("\n训练完成！")