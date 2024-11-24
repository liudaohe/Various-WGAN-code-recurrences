import argparse


def get_base_parser():
    """创建基础参数解析器"""
    parser = argparse.ArgumentParser(description="Various GAN Training")
    
    # 路径配置
    parser.add_argument("--output-dir", type=str, default="/",
                       help="Directory for saving results")
    parser.add_argument("--data-dir", type=str, default="/data",
                       help="Directory for datasets")

    # 基础参数
    parser.add_argument("--num-iterations", type=int, default=100000)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-interval", type=int, default=300,
                        help="Interval for saving checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--G", type=str, default="",
                        help="Path to generator checkpoint")
    parser.add_argument("--D", type=str, default="",
                        help="Path to discriminator checkpoint")
    parser.add_argument("--num_workers", type=int, default=2)

    return parser


def get_paras(gan_type):
    """根据GAN类型获取参数配置"""
    parser = get_base_parser()

    # GAN特定参数配置
    gan_configs = {
        "wgan-div": {
            "k": (2.0, "k parameter for WGAN-div"),
            "p": (6.0, "p parameter for WGAN-div"),
            "lr": (0.0002, None),
            "betas": ([0.5, 0.999], None)
        },
        "wgan-gp": {
            "k": (10.0, "k parameter for gradient penalty"),
            "lr": (0.0001, None),
            "betas": ([0.5, 0.9], None)
        },
        "wgan": {
            "lr": (0.00005, None),
            "c": (0.01, None)
        },
        "gan": {
            "lr": (0.0002, None),
            "momentum": (0.9, None)
        }
    }

    # 添加特定GAN的参数
    if gan_type in gan_configs:
        for param, (default, help_text) in gan_configs[gan_type].items():
            if param == "betas":
                parser.add_argument(f"--{param}", type=float, nargs=2,
                                    default=default, help=help_text)
            else:
                parser.add_argument(f"--{param}", type=float,
                                    default=default, help=help_text)

    return parser.parse_args()
