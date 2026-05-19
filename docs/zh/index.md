# Zipformer

Zipformer 是小米集团新一代 Kaldi 团队研发的新型语音编码器，它具有效果更好、计算更快、更省内存等诸多优点，是 google conformer 模型发布以后首个已知的在单数据集（Librispeech）上超越 conformer 论文的语音编码器，Zipformer 被 ICLR 2024 接收为 Oral 论文 (前 1.2%)。

## 概述

Zipformer 模型里面有众多的创新，主要的包括:

* 高效的模型结构：Downsampled encoder structure 和 Zipformer block
* 新 normalization：BiasNorm
* 新激活函数：Swoosh
* 新优化器：ScaledAdam 优化器
* 激活值限制策略：Balancer 和 Whitener

更多的细节请阅读[论文](https://arxiv.org/pdf/2310.11230.pdf), 中文用户也可以查阅我们的[博客](https://mp.weixin.qq.com/s/4N0xvA0RGG3IOPHPQ_vhZg)

## 快速开始

```bash
pip install zipformer
```

