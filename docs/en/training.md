---
comments: true
---

# Training

This page describes how to train a Zipformer speech recognition model, including data format, training scripts, and evaluation scripts.

## Data

This repository uses [atdataset](https://github.com/pkufool/ATdataset) as the dataloader. atdataset is a dataloader built on top of webdataset.


## Training

> The examples below use the medium model. For parameter settings of other variants, see the [Model Documentation](./models.md).

### Single-node multi-GPU

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

    To train a streaming model, simply add the `--causal 1` argument.

### Multi-node multi-GPU

!!! note

    Note that all nodes must have identical arguments except for `--world-size`, `--local-rank-start`, and `--local-world-size`.


Assume using 2 machines, each with 8 GPUs.

* First machine (assume IP is 127.0.0.3, serving as the master node)

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

* Second machine

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


## Evaluation

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


## Export

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
