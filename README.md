# Various WGAN Implementations

本仓库包含了多种WGAN（Wasserstein GAN）变体的PyTorch实现，包括：

- 原始GAN
- WGAN (Wasserstein GAN)
- WGAN-GP (WGAN with Gradient Penalty)
- WGAN-DIV (WGAN with Divergence)

## 项目结构

model/ # 模型定义
├── Discriminator.py # 判别器架构
└── Generator.py # 生成器架构

gan.py # 原始GAN训练
wgan.py # WGAN训练
wgan_gp.py # WGAN-GP训练
wgan_div.py # WGAN-DIV训练
utils.py # 工具函数
parameter.py # 参数配置
requirements.txt # 项目依赖

## 环境要求
bash
pip install -r requirements.txt
