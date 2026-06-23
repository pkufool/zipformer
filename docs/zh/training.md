---
comments: true
---

# 训练

本页介绍如何训练一个 zipformer 语音识别模型，包含数据的格式，训练和评测的脚本。

## 数据

本仓库使用 [atdataset](https://github.com/pkufool/ATdataset) 作为 dataloader， atdataset 是一个基于 webdataset 的数据加载器。


## 训练

> 下面的示例以 medium 模型为例，其他变体的的参数设置参见[模型文档](./models.md)

### 单机多卡

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

zipformer train \
    --world-size 8 \
    --exp-dir zipformer/exp_medium \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,384,512,384,256 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --bpe-model zh-en-8776 \
    --training-sets data/training_set.lst
    --num-epochs 20 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-cr-ctc 1 \
    --use-ctc 1 \
    --base-lr 0.045 \
    --use-transducer 1 \
    --use-attention-decoder 0 \
    --enable-spec-aug 0 \
    --ctc-loss-scale 0.2 \
    --cr-loss-scale 0.02 \
    --time-mask-ratio 2.5 \
    --lr-hours 50000 \
    --num-workers 2 \
    --max-duration 600
```

!!! note

    如果要训练流式模型，只需要增加 `--causal 1` 参数即可。

### 多机多卡

!!! note

    注意，所有 node 的参数，除了 `--world-size`, `--local-rank-start` 和 `--local-world-size` 之外，全部都必须一样。


假设使用 2 台机器，每台机器 8 卡。

* 第一台机器 (假设 ip 为 127.0.0.3, 作为 master 节点)

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

zipformer train \
    --world-size 16 \
    --master-addr 127.0.0.3 \
    --master-port 8808 \
    --local-rank-start 0 \
    --local-world-size 8 \
    --exp-dir zipformer/exp_medium \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,384,512,384,256 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --bpe-model zh-en-8776 \
    --training-sets data/training_set.lst
    --num-epochs 20 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-cr-ctc 1 \
    --use-ctc 1 \
    --base-lr 0.045 \
    --use-transducer 1 \
    --use-attention-decoder 0 \
    --enable-spec-aug 0 \
    --ctc-loss-scale 0.2 \
    --cr-loss-scale 0.02 \
    --time-mask-ratio 2.5 \
    --lr-hours 50000 \
    --num-workers 2 \
    --max-duration 600
```

* 第二台机器

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

zipformer train \
    --world-size 16 \
    --master-addr 127.0.0.3 \
    --master-port 8808 \
    --local-rank-start 8 \
    --local-world-size 8 \
    --exp-dir zipformer/exp_medium \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,384,512,384,256 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --bpe-model zh-en-8776 \
    --training-sets data/training_set.lst
    --num-epochs 20 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-cr-ctc 1 \
    --use-ctc 1 \
    --base-lr 0.045 \
    --use-transducer 1 \
    --use-attention-decoder 0 \
    --enable-spec-aug 0 \
    --ctc-loss-scale 0.2 \
    --cr-loss-scale 0.02 \
    --time-mask-ratio 2.5 \
    --lr-hours 50000 \
    --num-workers 2 \
    --max-duration 600
```


## 评测

```bash

zipformer decode \
    --exp-dir zipformer/exp_medium \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,384,512,384,256 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --epoch ITER \
    --avg AVG \
    --bpe-model zh-en-8776 \
    --test-sets test_clean,data/librispeech_test_clean.lst \
                test_other,data/librispeech_test_other.lst \
    --decoding-method rnnt-greedy-search

```


## 导出

```bash
zipformer export \
    --use-ctc 1 \
    --use-transducer 1 \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,384,512,384,256 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --exp-dir zipformer/exp_medium \
    --bpe-model zh-en-8776 \
    --iter ITER \
    --avg AVG
```