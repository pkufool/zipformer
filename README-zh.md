[English Version](./README.md)

<div align="center">

# zipformer

## A faster and better encoder for automatic speech recognition
</div>


## 概述

zipformer 是一个兼具性能和效率的语音编码器，针对语音识别任务特别优化，是在公平对比下唯一超过 google conformer 模型的语音识别模型。

### 特性

* 高效模型结构: Unet 类型的多层采样 encoder 及多种模块创新（BiasNorm, Swoosh, Balancer, Whitener）。
* 全新优化器：ScaledAdam。
* SOTA 性能且比 conformer 节省 50% FLOPs。
* 支持 CTC、Transducer 和 AED 建模。
* CR-CTC: 一致性正则化，打造更强的 CTC 模型。 


### 模型

zipformer ASR 模型目前提供 xlarge，large，medium，small 四种变体及对应的流式和非流式模型，下表为模型下载地址，更多具体信息请参见[文档](https://pkufool.github.io/zipformer/zh/models).

| 模型 | 参数量 | 配置 | ModelScope | Huggingface | 支持语言 | 支持架构 |
| -- | -- | -- | -- | -- | -- | -- |
| zipformer-xlarge           | 300M | --num-encoder-layers 2,2,4,5,4,2<br>--feedforward-dim 512,1024,2048,3072,2048,1024<br>--encoder-dim 192,384,768,1024,768,384<br>--encoder-unmasked-dim 192,256,320,512,320,256 | [link](https://www.modelscope.cn/models/pkufool/zipformer-xlarge) | [link](https://huggingface.co/pkufool/zipformer-xlarge) | 中文、英文 | CTC |
| zipformer-large            | 150M | --num-encoder-layers 2,2,4,5,4,2<br>--feedforward-dim 512,768,1536,2048,1536,768<br>--encoder-dim 192,256,512,768,512,256<br>--encoder-unmasked-dim 192,192,256,320,256,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-large) | [link](https://huggingface.co/pkufool/zipformer-large) | 中文、英文 | CTC、Transducer |
| zipformer-large-streaming  | 150M | --num-encoder-layers 2,2,4,5,4,2<br>--feedforward-dim 512,768,1536,2048,1536,768<br>--encoder-dim 192,256,512,768,512,256<br>--encoder-unmasked-dim 192,192,256,320,256,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-large-streaming) | [link](https://huggingface.co/pkufool/zipformer-large-streaming) | 中文、英文 | CTC、Transducer |
| zipformer-medium           | 65M | --num-encoder-layers 2,2,3,4,3,2<br>--feedforward-dim 512,768,1024,1536,1024,768<br>--encoder-dim 192,256,384,512,384,256<br>--encoder-unmasked-dim 192,192,256,256,256,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-medium) | [link](https://huggingface.co/pkufool/zipformer-medium) | 中文、英文 | CTC、Transducer  |
| zipformer-medium-streaming | 65M | --num-encoder-layers 2,2,3,4,3,2<br>--feedforward-dim 512,768,1024,1536,1024,768<br>--encoder-dim 192,256,384,512,384,256<br>--encoder-unmasked-dim 192,192,256,256,256,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-medium-streaming) | [link](https://huggingface.co/pkufool/zipformer-medium-streaming) | 中文、英文 | CTC、Transducer |
| zipformer-small            | 25M | --num-encoder-layers 2,2,2,2,2,2<br>--feedforward-dim 512,768,768,768,768,768<br>--encoder-dim 192,256,256,256,256,256<br>--encoder-unmasked-dim 192,192,192,192,192,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-small) | [link](https://huggingface.co/pkufool/zipformer-small) | 中文、英文 | CTC、Transducer |
| zipformer-small-streaming  | 25M | --num-encoder-layers 2,2,2,2,2,2<br>--feedforward-dim 512,768,768,768,768,768<br>--encoder-dim 192,256,256,256,256,256<br>--encoder-unmasked-dim 192,192,192,192,192,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-small-streaming) | [link](https://huggingface.co/pkufool/zipformer-small-streaming) | 中文、英文 | CTC、Transducer |

## 新闻

**2026/06/22:** 从 [icefall](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer) 创建 zipformer 独立仓库，并发布 xlarge、large、medium、small 中英文模型。


## 安装

```bash
pip install zipformer
```

## 用例

> [!TIP]
> 下面的示例采用非流式的 medium 模型，更多模型请查看[文档](https://pkufool.github.io/zipformer/zh/models)

### 命令行

```bash
# Use jit scripted model
# Transducer
zipformer inference --ms-model pkufool/zipformer-medium --model-type jit --ctc 0 en.wav zh.wav

# CTC
zipformer inference --ms-model pkufool/zipformer-medium --model-type jit --ctc 1 en.wav zh.wav

# Use onnx model
# Transducer
zipformer inference --ms-model pkufool/zipformer-medium --model-type onnx --ctc 0 en.wav zh.wav

# CTC
zipformer inference --ms-model pkufool/zipformer-medium --model-type onnx --ctc 1 en.wav zh.wav
```

### Python API

```python
from zipformer import inference

# jit scripted mdoel
result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='jit', ctc=False)

result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='jit', ctc=True)

# onnx model
result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='onnx', ctc=False)

result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='onnx', ctc=True)

# fp16 model
result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='onnx', ctc=False, dtype='fp16')

result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='onnx', ctc=True, dtype='fp16')
```

## 详细文档

更多关于模型训练、评测、部署的信息参见[文档](https://pkufool.github.io/zipformer/).


## 讨论 & 联系我们

有任务问题可以直接提 Issue [Github Issues](https://github.com/pkufool/zipformer/issues).

你也可以用微信扫描下面的二维码加入我们开发者群或者关注我们的微信公众号。

| 开发者群管理员 | 微信公众号 |
| ------------ | ----------------------- |
|![wechat](https://k2-fsa.org/assets/pic/wechat_group.jpg) |![wechat](https://k2-fsa.org/assets/pic/wechat_account.jpg) |


## 引用

```bibtex
@inproceedings{yao2024zipformer,
  title={Zipformer: A faster and better encoder for automatic speech recognition},
  author={Yao, Zengwei and Guo, Liyong and Yang, Xiaoyu and Kang, Wei and Kuang, Fangjun and Yang, Yifan and Jin, Zengrui and Lin, Long and Povey, Daniel},
  booktitle={International Conference on Learning Representations},
  volume={2024},
  pages={44440--44455},
  year={2024}
}

@inproceedings{yao2025cr,
  title={Cr-ctc: Consistency regularization on ctc for improved speech recognition},
  author={Yao, Zengwei and Kang, Wei and Yang, Xiaoyu and Kuang, Fangjun and Guo, Liyong and Zhu, Han and Jin, Zengrui and Li, Zhaoqing and Lin, Long and Povey, Daniel},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={26850--26868},
  year={2025}
}
```