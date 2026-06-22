<div align="right">

[中文](./README-zh.md)

</div>

<div align="center">

# zipformer

## A faster and better encoder for automatic speech recognition
</div>


## Overview

zipformer is a speech encoder that achieves both high performance and efficiency. It is specifically optimized for speech recognition tasks and is the only model that outperforms Google's Conformer under fair comparison.

### Features

* Efficient model architecture: UNet-style multi-scale encoder with module innovations (BiasNorm, Swoosh, Balancer, Whitener).
* New optimizer: ScaledAdam.
* State-of-the-art performance with 50% fewer FLOPs than Conformer.
* Supports CTC, Transducer, and AED modeling.
* CR-CTC: Consistency regularization for stronger CTC models.


### Models

zipformer ASR models are available in xlarge, large, medium, and small variants, with both streaming and non-streaming versions. The table below provides download links. For more details, please refer to the [documentation](https://pkufool.github.io/zipformer/models).

| Model | Parameters | Config | ModelScope | Huggingface | Languages | Architectures |
| -- | -- | -- | -- | -- | -- | -- |
| zipformer-xlarge           | 300M | --num-encoder-layers 2,2,4,5,4,2<br>--feedforward-dim 512,1024,2048,3072,2048,1024<br>--encoder-dim 192,384,768,1024,768,384<br>--encoder-unmasked-dim 192,256,320,512,320,256 | [link](https://www.modelscope.cn/models/pkufool/zipformer-xlarge) | [link](https://huggingface.co/pkufool/zipformer-xlarge) | Chinese, English | CTC |
| zipformer-large            | 150M | --num-encoder-layers 2,2,4,5,4,2<br>--feedforward-dim 512,768,1536,2048,1536,768<br>--encoder-dim 192,256,512,768,512,256<br>--encoder-unmasked-dim 192,192,256,320,256,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-large) | [link](https://huggingface.co/pkufool/zipformer-large) | Chinese, English | CTC, Transducer |
| zipformer-large-streaming  | 150M | --num-encoder-layers 2,2,4,5,4,2<br>--feedforward-dim 512,768,1536,2048,1536,768<br>--encoder-dim 192,256,512,768,512,256<br>--encoder-unmasked-dim 192,192,256,320,256,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-large-streaming) | [link](https://huggingface.co/pkufool/zipformer-large-streaming) | Chinese, English | CTC, Transducer |
| zipformer-medium           | 65M | --num-encoder-layers 2,2,3,4,3,2<br>--feedforward-dim 512,768,1024,1536,1024,768<br>--encoder-dim 192,256,384,512,384,256<br>--encoder-unmasked-dim 192,192,256,256,256,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-medium) | [link](https://huggingface.co/pkufool/zipformer-medium) | Chinese, English | CTC, Transducer  |
| zipformer-medium-streaming | 65M | --num-encoder-layers 2,2,3,4,3,2<br>--feedforward-dim 512,768,1024,1536,1024,768<br>--encoder-dim 192,256,384,512,384,256<br>--encoder-unmasked-dim 192,192,256,256,256,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-medium-streaming) | [link](https://huggingface.co/pkufool/zipformer-medium-streaming) | Chinese, English | CTC, Transducer |
| zipformer-small            | 25M | --num-encoder-layers 2,2,2,2,2,2<br>--feedforward-dim 512,768,768,768,768,768<br>--encoder-dim 192,256,256,256,256,256<br>--encoder-unmasked-dim 192,192,192,192,192,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-small) | [link](https://huggingface.co/pkufool/zipformer-small) | Chinese, English | CTC, Transducer |
| zipformer-small-streaming  | 25M | --num-encoder-layers 2,2,2,2,2,2<br>--feedforward-dim 512,768,768,768,768,768<br>--encoder-dim 192,256,256,256,256,256<br>--encoder-unmasked-dim 192,192,192,192,192,192 | [link](https://www.modelscope.cn/models/pkufool/zipformer-small-streaming) | [link](https://huggingface.co/pkufool/zipformer-small-streaming) | Chinese, English | CTC, Transducer |

## News

**2026/06/22:** Created standalone zipformer repository from [icefall](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer), and released xlarge, large, medium, and small Chinese/English models.


## Installation

```bash
pip install zipformer
```

## Usage

> [!TIP]
> The examples below use the non-streaming medium model. For more models, please refer to the [documentation](https://pkufool.github.io/zipformer/models).

### Command Line

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

# jit scripted model
result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='jit', ctc=False)

result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='jit', ctc=True)

# onnx model
result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='onnx', ctc=False)

result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='onnx', ctc=True)

# fp16 model
result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='onnx', ctc=False, dtype='fp16')

result = inference([en.wav, zh.wav], ms_model='pkufool/zipformer-medium', model_type='onnx', ctc=True, dtype='fp16')
```

## Documentation

For more information about model training, evaluation, and deployment, please refer to the [documentation](https://pkufool.github.io/zipformer/).


## Discussion & Contact

For task-related issues, please open an issue on [GitHub Issues](https://github.com/pkufool/zipformer/issues).

You can also scan the QR code below to join our developer WeChat group or follow our WeChat official account.

| Developer Group Admin | WeChat Official Account |
| ------------ | ----------------------- |
|![wechat](https://k2-fsa.org/assets/pic/wechat_group.jpg) |![wechat](https://k2-fsa.org/assets/pic/wechat_account.jpg) |


## Citation

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
