# Various WGAN Implementations

本仓库包含了多种WGAN（Wasserstein GAN）变体的PyTorch实现，包括：

- 原始GAN
- WGAN (Wasserstein GAN)
- WGAN-GP (WGAN with Gradient Penalty)
- WGAN-DIV (WGAN with Divergence)

## 项目结构

```text
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
```
## 环境要求
```
pip install -r requirements.txt
```
## 训练参数配置

### 基础参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| output-dir | str | "/" | 结果保存目录 |
| data-dir | str | "/data" | 数据集目录 |
| num-iterations | int | 100000 | 训练迭代次数 |
| n-critic | int | 5 | 判别器训练次数 |
| batch-size | int | 64 | 批次大小 |
| checkpoint-interval | int | 300 | 检查点保存间隔 |
| device | str | "cuda" | 训练设备 |
| G | str | "" | 生成器检查点路径 |
| D | str | "" | 判别器检查点路径 |
| num_workers | int | 2 | 数据加载线程数 |

## 训练参数配置

### 基础参数（所有GAN通用）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| output-dir | str | "/" | 结果保存目录 |
| data-dir | str | "/data" | 数据集目录 |
| num-iterations | int | 100000 | 训练迭代次数 |
| n-critic | int | 5 | 判别器训练次数 |
| batch-size | int | 64 | 批次大小 |
| checkpoint-interval | int | 300 | 检查点保存间隔 |
| device | str | "cuda" | 训练设备 |
| G | str | "" | 生成器检查点路径 |
| D | str | "" | 判别器检查点路径 |
| num_workers | int | 2 | 数据加载线程数 |

### GAN特定参数

#### 原始GAN
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| lr | float | 0.0002 | 学习率 |
| momentum | float | 0.9 | SGD动量系数 |

#### WGAN
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| lr | float | 0.00005 | RMSprop学习率 |
| c | float | 0.01 | 权重裁剪参数 |

#### WGAN-GP
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| lr | float | 0.0001 | Adam学习率 |
| betas | float[2] | [0.5, 0.9] | Adam优化器参数 |
| k | float | 10.0 | 梯度惩罚系数 |

#### WGAN-DIV
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| lr | float | 0.0002 | Adam学习率 |
| betas | float[2] | [0.5, 0.999] | Adam优化器参数 |
| k | float | 2.0 | WGAN-div的k参数 |
| p | float | 6.0 | WGAN-div的p参数 |


## 运行

```
python wgan_div
```

## 判别器(Discriminator)，WGAN系列中称为（Critic）结构
```
Input (3×64×64)
    ↓
Conv2d(3→64, 3×3, s=1)
    ↓
ResBlock1: 64→128 + Downsample
    ├── Main: Conv(64→128, 3×3) → BN → LeakyReLU → Conv(128→128, 3×3) → BN → LeakyReLU → Conv(128→128, 3×3, s=2) → BN
    └── Bypass: Conv(64→128, 1×1, s=2) → BN
    ↓
ResBlock2: 128→256 + Downsample
    ├── Main: Conv(128→256, 3×3) → BN → LeakyReLU → Conv(256→256, 3×3) → BN → LeakyReLU → Conv(256→256, 3×3, s=2) → BN
    └── Bypass: Conv(128→256, 1×1, s=2) → BN
    ↓
ResBlock3: 256→512 + Downsample
    ├── Main: Conv(256→512, 3×3) → BN → LeakyReLU → Conv(512→512, 3×3) → BN → LeakyReLU → Conv(512→512, 3×3, s=2) → BN
    └── Bypass: Conv(256→512, 1×1, s=2) → BN
    ↓
ResBlock4: 512→512 + Downsample
    ├── Main: Conv(512→512, 3×3) → BN → LeakyReLU → Conv(512→512, 3×3) → BN → LeakyReLU → Conv(512→512, 3×3, s=2) → BN
    └── Bypass: Conv(512→512, 1×1, s=2) → BN
    ↓
Flatten
    ↓
Linear(512×4×4 → 1)
    ↓
Output (1)
```

## 生成器(Generator)结构
```
Input (z_dim=128)
    ↓
Linear(128 → 512×4×4) → BN → ReLU
    ↓
Reshape(512×4×4)
    ↓
ResBlock1: 512→512 + Upsample
    ├── Main: Conv(512→512, 3×3) → BN → ReLU → Conv(512→512, 3×3) → BN → ReLU → ConvTranspose(512→512, 4×4, s=2) → BN
    └── Bypass: ConvTranspose(512→512, 4×4, s=2) → BN
    ↓
ResBlock2: 512→256 + Upsample
    ├── Main: Conv(512→256, 3×3) → BN → ReLU → Conv(256→256, 3×3) → BN → ReLU → ConvTranspose(256→256, 4×4, s=2) → BN
    └── Bypass: ConvTranspose(512→256, 4×4, s=2) → BN
    ↓
ResBlock3: 256→128 + Upsample
    ├── Main: Conv(256→128, 3×3) → BN → ReLU → Conv(128→128, 3×3) → BN → ReLU → ConvTranspose(128→128, 4×4, s=2) → BN
    └── Bypass: ConvTranspose(256→128, 4×4, s=2) → BN
    ↓
ResBlock4: 128→64 + Upsample
    ├── Main: Conv(128→64, 3×3) → BN → ReLU → Conv(64→64, 3×3) → BN → ReLU → ConvTranspose(64→64, 4×4, s=2) → BN
    └── Bypass: ConvTranspose(128→64, 4×4, s=2) → BN
    ↓
Conv2d(64→3, 3×3, s=1)
    ↓
Tanh
    ↓
Output (3×64×64)

```












## 参考文献

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
  - Goodfellow, I., et al. (2014)
  - 原始GAN论文

- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
  - Arjovsky, M., et al. (2017)
  - 提出了Wasserstein距离作为GAN的损失函数

- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
  - Gulrajani, I., et al. (2017)
  - 提出了梯度惩罚（GP）来改进WGAN训练

- [Wasserstein Divergence for GANs](https://arxiv.org/abs/1712.01026)
  - Wu, J., et al. (2017)
  - 提出了WGAN-DIV，使用Wasserstein散度
