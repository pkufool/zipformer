



### xlarge ctc

```bash
python ./zipformer/train_large.py \
        --world-size 8 \
        --master-port 12361 \
        --exp-dir zipformer/exp_80w_cr_ctc \
        --num-encoder-layers 2,2,4,5,4,2 \
        --feedforward-dim 512,1024,2048,3072,2048,1024 \
        --encoder-dim 192,384,768,1024,768,384 \
        --encoder-unmasked-dim 192,256,320,512,320,256 \
        --bpe-model zh-en-yue-11661 \
        --num-epochs 20 \
        --use-fp16 1 \
        --start-epoch 4 \
        --use-cr-ctc 1 \
        --use-ctc 1 \
        --base-lr 0.035 \
        --use-transducer 0 \
        --use-attention-decoder 0 \
        --enable-spec-aug 0 \
        --cr-loss-scale 0.2 \
        --time-mask-ratio 2.5 \
        --lr-hours 100000 \
        --num-workers 2 \
        --max-duration 600
```