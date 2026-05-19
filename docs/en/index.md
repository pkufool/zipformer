# Zipformer

Zipformer is a novel speech encoder developed by the Next-Gen Kaldi team at Xiaomi. It offers superior accuracy, faster computation, and lower memory usage. It is the first known speech encoder to surpass the Conformer paper on a single dataset (LibriSpeech) after Google's Conformer was published. Zipformer was accepted as an Oral paper (top 1.2%) at ICLR 2024.

## Overview

Zipformer introduces numerous innovations, including:

* Efficient model architecture: Downsampled encoder structure and Zipformer block
* New normalization: BiasNorm
* New activation function: Swoosh
* New optimizer: ScaledAdam
* Activation value limiting strategies: Balancer and Whitener

For more details, please refer to the [paper](https://arxiv.org/pdf/2310.11230.pdf). Chinese readers can also check our [blog post](https://mp.weixin.qq.com/s/4N0xvA0RGG3IOPHPQ_vhZg).

## Quick Start

```bash
pip install zipformer
```
