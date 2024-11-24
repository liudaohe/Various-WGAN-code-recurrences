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
    n_critic, batch_size, lr, num_iterations, device, betas, checkpoint_interval, num_workers,OUTPUT_DIR, DATA_DIR = \
        args.n_critic, args.batch_size, args.lr, args.num_iterations, args.device, \
        args.betas, args.checkpoint_interval, args.num_workers, args.output_dir, args.data_dir

    dataloader = utils.load_data(batch_size, num_workers)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    G = Generator().to(device)
    D = Discriminator(sig=True).to(device)

    fid = 0
    print(f"""
    初始化训练...
    训练参数:
    ├── 模型参数:
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
    for i, (real_imgs, _) in enumerate(dataloader):
        for j, img in enumerate(real_imgs):
            vutils.save_image(
                img,
                f"{real_temp_dir}/{i}_{j}.png",
                normalize=True,
                value_range=(-1, 1)
            )
        if i * len(real_imgs) >= 5000:  # 保存足够的图像后停止
            break

    G_losses = []
    D_losses = []
    FIDs = []
    grad_flows = []

    print("\n开始训练循环...")
    for iteration in tqdm(range(num_iterations)):

        for _ in range(n_critic):

            # 步骤2：采样真实数据
            try:
                real_data = next(iter(dataloader))[0].to(device)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                real_data = next(dataloader_iterator)[0].to(device)

            batch_size = real_data.size(0)

            # 步骤3：采样高斯噪声
            z = torch.randn(batch_size, 128).to(device)

            # 生成假数据
            fake_data = G(z)


            optimizerD.zero_grad()

            d_real = D(real_data)
            d_fake = D(fake_data.detach())

            d_loss = -(torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8)))
            d_loss.backward()
            optimizerD.step()
            D_losses.append(d_loss.item())
            grad_flow_d = torch.cat([p.grad.abs().mean().unsqueeze(0)
                                     for p in D.parameters() if p.grad is not None])

        # 步骤6：更新生成器
        optimizerG.zero_grad()
        z = torch.randn(batch_size, 128).to(device)
        fake_data = G(z)
        g_fake = D(fake_data)
        g_loss = -torch.mean(torch.log(g_fake + 1e-8))

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
    args = get_paras("gan")

    utils.set_seed()
    print("数据集加载完成")
    device = utils.choice_device()

    print(f"模型已初始化并移至{device}")
    train(args)

    print("\n训练完成！")