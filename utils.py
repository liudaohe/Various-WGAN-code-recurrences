
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
import numpy as np
import shutil
from contextlib import contextmanager

DATA_DIR = "/data"

@contextmanager
def temp_dir(path):
    """创建临时目录的上下文管理器"""
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        if os.path.exists(path):
            shutil.rmtree(path)



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


#加载数据集
def load_data(batch_size,num_workers):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                         download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
#选择设备
def choice_device():    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#保存检查点
def save_checkpoint(save_dir, iteration, G, D, optimizerG, optimizerD,
                    G_losses, D_losses, FIDs, grad_flows, filename='last_checkpoint.pt'):
    """
    保存训练检查点

    Args:
        save_dir (str): 保存目录
        iteration (int): 当前迭代次数
        G: 生成器模型
        D: 判别器模型
        optimizerG: 生成器优化器
        optimizerD: 判别器优化器
        G_losses (list): 生成器损失历史
        D_losses (list): 判别器损失历史
        FIDs (list): FID分数历史
        grad_flows (list): 梯度流历史
        filename (str): 保存的文件名

    Returns:
        bool: 保存是否成功
    """
    try:
        checkpoint_path = os.path.join(save_dir, filename)
        torch.save({
            'iteration': iteration,
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'optimizerG_state': optimizerG.state_dict(),
            'optimizerD_state': optimizerD.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
            'FIDs': FIDs,
            'grad_flows': grad_flows
        }, checkpoint_path)

        print(f"\n检查点保存状态:")
        print(f"保存路径: {checkpoint_path}")
        print(f"文件是否存在: {os.path.exists(checkpoint_path)}")
        if os.path.exists(checkpoint_path):
            print(f"文件大小: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print(f"保存检查点时发生错误: {str(e)}")
        return False


def compute_fid_and_save_samples(G, device, real_temp_dir, samples_dir, iteration,
                                 batch_size=64, n_fid_samples=5000, n_display_samples=64):
    """计算FID分数并保存样本图像"""
    # 使用no_grad上下文管理器
    with torch.no_grad(), temp_dir(f'/image/fake_iter_{iteration}') as fake_temp_dir:
        try:
            # 生成图像用于FID计算
            n_gen = 0
            while n_gen < n_fid_samples:
                z = torch.randn(batch_size, 128).to(device)
                fake_imgs = G(z)
                for j, img in enumerate(fake_imgs):
                    vutils.save_image(
                        img,
                        f"{fake_temp_dir}/{n_gen + j}.png",
                        normalize=True,
                        value_range=(-1, 1)
                    )
                    n_gen += 1
                    if n_gen >= n_fid_samples:
                        break

            # 计算FID
            fid = fid_score.calculate_fid_given_paths(
                [real_temp_dir, fake_temp_dir],
                batch_size=50,
                device=device,
                dims=2048
            )

            # 保存展示用的样本
            fake_images = G(torch.randn(n_display_samples, 128).to(device))
            vutils.save_image(
                fake_images,
                os.path.join(samples_dir, f'生成样本_迭代_{iteration}.png'),
                normalize=True,
                value_range=(-1, 1)
            )

            return float(fid)

        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            return float('inf')