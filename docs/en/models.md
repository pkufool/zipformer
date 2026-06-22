# Speech Recognition Models

This page lists zipformer pre-trained models and their performance on commonly used open-source test sets, along with usage instructions.

## Chinese-English Models

### Non-Streaming Models

| Name | Parameters | Download | aishell test 1 / 2 |  wenetspeech test-net/meeting | Common Voice zh | kespeech test | librispeech test-clean / other | gigaspeech test | Common voice en | tedium test |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| xlarge-ctc | 300M | [HuggingFace](https://huggingface.co/pkufool/zipformer-xlarge)  | 1.61 / 2.7  | 5.35 / 6.39 | 8.26 | 5.74 | 3.51 / 7.78 | 14.53 | 28.57 | 15.07 |
| large-ctc  | 150M  | [HuggingFace](https://huggingface.co/pkufool/zipformer-large)  | 2.51 / 3.51 | 6.23 / 6.67 | 7.96 | 8.95 | 2.62 / 5.17 | 10.73 | 12.99 | 10.11 |
| large-rnnt | 150M | [HuggingFace](https://huggingface.co/pkufool/zipformer-large)   | 2.42 / 3.55 | 6.7 / 7.81  | 7.92 | 8.88 | 2.27 / 4.64 | 10.08 | 11.27 | 9.82  |
| medium-ctc | 65M  | [HuggingFace](https://huggingface.co/pkufool/zipformer-medium)  | 3.08 / 3.98 | 7.08 / 7.62 | 9.2  | 11.23| 3.01 / 6.06 | 11.22 | 15.28 | 10.38 |
| medium-rnnt| 65M  | [HuggingFace](https://huggingface.co/pkufool/zipformer-medium)  | 2.67 / 3.67 | 6.79 / 7.33 | 8.97 | 10.67| 2.61 / 5.36 | 10.56 | 12.94 | 10.06 |
| small-ctc  | 35M  | [HuggingFace](https://huggingface.co/pkufool/zipformer-small)   | 4.82 / 5.5  | 10.09 / 11.3| 12.76|16.07 | 5.12 / 10.67|22.27  | 23.7  | 11.04 |
| small-rnnt | 35M  | [HuggingFace](https://huggingface.co/pkufool/zipformer-small)   | 3.92 / 4.74 | 9.09 / 10.57|11.86 | 14.84|3.78 / 8.65  | 16.1  | 18.21 | 6.79  |

#### Directory Structure

Below are the files included in non-streaming zipformer speech recognition models:

```
.
├── ctc.fp16.onnx
├── ctc.int8.onnx
├── ctc.onnx
├── data
│   ├── tokens.txt
│   └── zh-en-8776.vocab
├── decoder.onnx
├── encoder.fp16.onnx
├── encoder.int8.onnx
├── encoder.onnx
├── jit_model.pt
├── joiner.fp16.onnx
├── joiner.int8.onnx
├── joiner.onnx
└── model.pt
```

* `model.pt` is the PyTorch `state_dict` of the model, shared by CTC and Transducer models. It can be used to export jit scripted and ONNX models, or as a starting point for fine-tuning.
* `jit_model.pt` is the jit scripted model, shared by CTC and Transducer models, suitable for deployment with torch.jit.script.
* `ctc.onnx, ctc.fp16.onnx, ctc.int8.onnx` are the exported CTC head ONNX models, corresponding to float32, float16, and int8 data types respectively (note: some models may not include fp16 and int8 variants).
* `encoder.onnx, encoder.fp16.onnx, encoder.int8.onnx` are the exported Transducer encoder ONNX models, corresponding to float32, float16, and int8 data types respectively (note: some models may not include fp16 and int8 variants). `decoder.onnx` is the exported Transducer decoder ONNX model (note: decoder does not have fp16 and int8 variants). `joiner.onnx, joiner.fp16.onnx, joiner.int8.onnx` are the exported Transducer joiner ONNX models, corresponding to float32, float16, and int8 data types respectively (note: some models may not include fp16 and int8 variants).
* The data directory contains the BPE model and tokens.


#### Usage

This section only covers usage methods supported by the [zipformer repository](https://github.com/pkufool/zipformer) (i.e., Python-based usage). For inference and deployment on other languages, operating systems, and hardware platforms, please refer to the [deployment section](./deployment.md).

##### Command Line

> The examples below use zipformer-large; other models follow the same pattern.

* CTC head inference

=== "Using downloaded models"

    ```bash
    # jit script model
    zipformer inference \
        --model zipformer-large/jit_model.pt \
        --ctc 1 \
        --model-type jit \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx model
    zipformer inference \
        --model zipformer-large/ctc.onnx \
        --ctc 1 \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx fp16 model
    zipformer inference \
        --model zipformer-large/ctc.fp16.onnx \
        --ctc 1 \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav
    
    # onnx int8 model
    zipformer inference \
        --model zipformer-large/ctc.int8.onnx \
        --ctc 1 \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav
    ```

=== "Auto-downloading from HuggingFace"

    ```bash
    # jit script model
    zipformer inference \
        --hf-model pkufool/zipformer-large \
        --ctc 1 \
        --model-type jit \
        data/en.wav data/zh.wav

    # onnx model
    zipformer inference \
        --hf-model pkufool/zipformer-large \
        --ctc 1 \
        --model-type onnx \
        data/en.wav data/zh.wav

    # onnx fp16 model
    zipformer inference \
        --hf-model pkufool/zipformer-large \
        --ctc 1 \
        --dtype fp16 \
        --model-type onnx \
        data/en.wav data/zh.wav
    
    # onnx int8 model
    zipformer inference \
        --hf-model pkufool/zipformer-large \
        --ctc 1 \
        --dtype int8 \
        --model-type onnx \
        data/en.wav data/zh.wav
    ```

* Transducer head inference

=== "Using downloaded models"

    ```bash
    # jit script model
    zipformer inference \
        --model zipformer-large/jit_model.pt \
        --model-type jit \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx model
    zipformer inference \
        --encoder zipformer-large/encoder.onnx \
        --decoder zipformer-large/decoder.onnx \
        --joiner zipformer-large/joiner.onnx \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx fp16 model
    zipformer inference \
        --encoder zipformer-large/encoder.fp16.onnx \
        --decoder zipformer-large/decoder.onnx \
        --joiner zipformer-large/joiner.fp16.onnx \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx int8 model
    zipformer inference \
        --encoder zipformer-large/encoder.int8.onnx \
        --decoder zipformer-large/decoder.onnx \
        --joiner zipformer-large/joiner.int8.onnx \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav
    ```

=== "Auto-downloading from HuggingFace"

    ```bash
    # jit script model
    zipformer inference \
        --hf-model pkufool/zipformer-large \
        --model-type jit \
        data/en.wav data/zh.wav

    # onnx model
    zipformer inference \
        --hf-model pkufool/zipformer-large \
        --model-type onnx \
        data/en.wav data/zh.wav

    # onnx fp16 model
    zipformer inference \
        --hf-model pkufool/zipformer-large \
        --dtype fp16 \
        --model-type onnx \
        data/en.wav data/zh.wav
    
    # onnx int8 model
    zipformer inference \
        --hf-model pkufool/zipformer-large \
        --dtype int8 \
        --model-type onnx \
        data/en.wav data/zh.wav
    ```

##### Python API

* CTC head inference

=== "Using downloaded models"

    ```python
    from zipformer import inference

    # jit script model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="jit",
        model="zipformer-large/jit_model.pt",
        tokens="data/tokens.txt",
        ctc=True,
    )

    # onnx model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        model="zipformer-large/ctc.onnx",
        tokens="data/tokens.txt",
        ctc=True,
    )

    # onnx fp16 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        model="zipformer-large/ctc.fp16.onnx",
        tokens="data/tokens.txt",
        ctc=True,
    )

    # onnx int8 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        model="zipformer-large/ctc.int8.onnx",
        tokens="data/tokens.txt",
        ctc=True,
    )
    ```

=== "Auto-downloading from HuggingFace"

    ```python
    from zipformer import inference

    # jit script model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="jit",
        hf_model="pkufool/zipformer-large",
        ctc=True,
    )

    # onnx model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large",
        ctc=True,
    )

    # onnx fp16 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large",
        ctc=True,
        dtype="fp16",
    )

    # onnx int8 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large",
        ctc=True,
        dtype="int8",
    )
    ```

* Transducer head inference

=== "Using downloaded models"

    ```python
    from zipformer import inference

    # jit script model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="jit",
        model="zipformer-large/jit_model.pt",
        tokens="data/tokens.txt",
    )

    # onnx model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        encoder="zipformer-large/encoder.onnx",
        decoder="zipformer-large/decoder.onnx",
        joiner="zipformer-large/joiner.onnx",
        tokens="data/tokens.txt",
    )

    # onnx fp16 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        encoder="zipformer-large/encoder.fp16.onnx",
        decoder="zipformer-large/decoder.onnx",
        joiner="zipformer-large/joiner.fp16.onnx",
        tokens="data/tokens.txt",
    )

    # onnx int8 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        encoder="zipformer-large/encoder.int8.onnx",
        decoder="zipformer-large/decoder.onnx",
        joiner="zipformer-large/joiner.int8.onnx",
        tokens="data/tokens.txt",
    )
    ```

=== "Auto-downloading from HuggingFace"

    ```python
    from zipformer import inference

    # jit script model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="jit",
        hf_model="pkufool/zipformer-large",
    )

    # onnx model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large",
    )

    # onnx fp16 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large",
        dtype="fp16",
    )

    # onnx int8 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large",
        dtype="int8",
    )
    ```

### Streaming Models

> The evaluation results below use --chunk-size 16 --left-context-frames 128

| Name | Parameters | Download | aishell test 1 / 2 |  wenetspeech test-net/meeting | Common Voice zh | kespeech test | librispeech test-clean / other | gigaspeech test | Common voice en | tedium test |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| large-ctc   | 150M | [HuggingFace](https://huggingface.co/pkufool/zipformer-large-streaming)   | 3.78 / 4.71 | 8.65 / 10.54 | 11.8 | 15.35 | 3.74 / 8.5 | 12.32 | 19.7 | 10.92 |
| large-rnnt  | 150M | [HuggingFace](https://huggingface.co/pkufool/zipformer-large-streaming)   | 3.53 / 4.48 | 8.31 / 10.27 | 11.99| 14.83 | 3.26 / 7.51 | 11.77| 17.53| 10.82 |
| medium-ctc  | 65M  | [HuggingFace](https://huggingface.co/pkufool/zipformer-medium-streaming)  | 4.46 / 5.09 | 9.74 / 11.21 | 12.68| 11.26 | 4.28 / 9.4 | 12.96 | 21.77| 11.26 |
| medium-rnnt | 65M  | [HuggingFace](https://huggingface.co/pkufool/zipformer-medium-streaming)  | 3.9 / 4.79  | 9.05 / 10.82 | 12.41| 17.89 | 3.64 / 8.08 | 12.13| 18.97| 10.9  |
| small-ctc   | 35M  | [HuggingFace](https://huggingface.co/pkufool/zipformer-small-streaming)   | 6.7 / 7.24  | 12.92 / 16.45| 17.18| 23.32 | 19.4 / 29.66| 26.18| 33.52| 17.67 |
| small-rnnt  | 35M  | [HuggingFace](https://huggingface.co/pkufool/zipformer-small-streaming)   | 5.69 / 6.26 | 12.06 / 16.13| 16.51| 22.29 | 8.15 / 16.91| 19.77| 28.54| 14.23 |


#### Directory Structure

Below are the files included in streaming zipformer speech recognition models:

```
.
├── ctc-chunk-16-left-64.fp16.onnx
├── ctc-chunk-16-left-64.int8.onnx
├── ctc-chunk-16-left-64.onnx
├── ctc-chunk-32-left-128.fp16.onnx
├── ctc-chunk-32-left-128.int8.onnx
├── ctc-chunk-32-left-128.onnx
├── ctc-chunk-64-left-256.fp16.onnx
├── ctc-chunk-64-left-256.int8.onnx
├── ctc-chunk-64-left-256.onnx
├── data
│   ├── tokens.txt
│   └── zh-en-8776.vocab
├── decoder-chunk-16-left-64.onnx
├── decoder-chunk-32-left-128.onnx
├── decoder-chunk-64-left-256.onnx
├── encoder-chunk-16-left-64.fp16.onnx
├── encoder-chunk-16-left-64.int8.onnx
├── encoder-chunk-16-left-64.onnx
├── encoder-chunk-32-left-128.fp16.onnx
├── encoder-chunk-32-left-128.int8.onnx
├── encoder-chunk-32-left-128.onnx
├── encoder-chunk-64-left-256.fp16.onnx
├── encoder-chunk-64-left-256.int8.onnx
├── encoder-chunk-64-left-256.onnx
├── jit_model-chunk-16-left-64.pt
├── jit_model-chunk-32-left-128.pt
├── jit_model-chunk-64-left-256.pt
├── joiner-chunk-16-left-64.fp16.onnx
├── joiner-chunk-16-left-64.int8.onnx
├── joiner-chunk-16-left-64.onnx
├── joiner-chunk-32-left-128.fp16.onnx
├── joiner-chunk-32-left-128.int8.onnx
├── joiner-chunk-32-left-128.onnx
├── joiner-chunk-64-left-256.fp16.onnx
├── joiner-chunk-64-left-256.int8.onnx
├── joiner-chunk-64-left-256.onnx
└── model.pt
```

> The exported streaming models come in three latency variants: `chunk-size=16, left-context-frames=64` (320ms latency), `chunk-size=32, left-context-frames=128` (640ms latency), `chunk-size=64, left-context-frames=256` (1280ms latency).

* `model.pt` is the PyTorch `state_dict` of the model, shared by CTC and Transducer models. It can be used to export jit scripted and ONNX models, or as a starting point for fine-tuning.
* `jit_model-chunk-{chunk-size}-left-{left-context-frames}.pt` is the jit scripted model, shared by CTC and Transducer models, suitable for deployment with torch.jit.script.
* `ctc-chunk-{chunk-size}-left-{left-context-frames}.onnx, ctc-chunk-{chunk-size}-left-{left-context-frames}.fp16.onnx, ctc-chunk-{chunk-size}-left-{left-context-frames}.int8.onnx` are the exported CTC head ONNX models, corresponding to float32, float16, and int8 data types respectively (note: some models may not include fp16 and int8 variants).
* `encoder-chunk-{chunk-size}-left-{left-context-frames}.onnx, encoder-chunk-{chunk-size}-left-{left-context-frames}.fp16.onnx, encoder-chunk-{chunk-size}-left-{left-context-frames}.int8.onnx` are the exported Transducer encoder ONNX models, corresponding to float32, float16, and int8 data types respectively (note: some models may not include fp16 and int8 variants). `decoder-chunk-{chunk-size}-left-{left-context-frames}.onnx` is the exported Transducer decoder ONNX model (note: decoder does not have fp16 and int8 variants). `joiner-chunk-{chunk-size}-left-{left-context-frames}.onnx, joiner-chunk-{chunk-size}-left-{left-context-frames}.fp16.onnx, joiner-chunk-{chunk-size}-left-{left-context-frames}.int8.onnx` are the exported Transducer joiner ONNX models, corresponding to float32, float16, and int8 data types respectively (note: some models may not include fp16 and int8 variants).
* The data directory contains the BPE model and tokens.

#### Usage

This section only covers usage methods supported by the [zipformer repository](https://github.com/pkufool/zipformer) (i.e., Python-based usage). For inference and deployment on other languages, operating systems, and hardware platforms, please refer to the [deployment section](./deployment.md).

##### Command Line

> The examples below use zipformer-large-streaming with `chunk-size=32, left-context-frames=128`; other models follow the same pattern.

* CTC head inference

=== "Using downloaded models"

    ```bash
    # jit script model
    zipformer inference \
        --model zipformer-large-streaming/jit_model-chunk-32-left-128.pt \
        --ctc 1 \
        --streaming 1 \
        --model-type jit \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx model
    zipformer inference \
        --model zipformer-large-streaming/ctc-chunk-32-left-128.onnx \
        --ctc 1 \
        --streaming 1 \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx fp16 model
    zipformer inference \
        --model zipformer-large-streaming/ctc-chunk-32-left-128.fp16.onnx \
        --ctc 1 \
        --streaming 1 \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx int8 model
    zipformer inference \
        --model zipformer-large-streaming/ctc-chunk-32-left-128.int8.onnx \
        --ctc 1 \
        --streaming 1 \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav
    ```

=== "Auto-downloading from HuggingFace"

    ```bash
    # jit script model
    zipformer inference \
        --hf-model pkufool/zipformer-large-streaming \
        --ctc 1 \
        --streaming 1 \
        --model-type jit \
        data/en.wav data/zh.wav

    # onnx model
    zipformer inference \
        --hf-model pkufool/zipformer-large-streaming \
        --ctc 1 \
        --streaming 1 \
        --model-type onnx \
        data/en.wav data/zh.wav

    # onnx fp16 model
    zipformer inference \
        --hf-model pkufool/zipformer-large-streaming \
        --ctc 1 \
        --streaming 1 \
        --dtype fp16 \
        --model-type onnx \
        data/en.wav data/zh.wav

    # onnx int8 model
    zipformer inference \
        --hf-model pkufool/zipformer-large-streaming \
        --ctc 1 \
        --streaming 1 \
        --dtype int8 \
        --model-type onnx \
        data/en.wav data/zh.wav
    ```

* Transducer head inference

=== "Using downloaded models"

    ```bash
    # jit script model
    zipformer inference \
        --model zipformer-large-streaming/jit_model-chunk-32-left-128.pt \
        --streaming 1 \
        --model-type jit \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx model
    zipformer inference \
        --encoder zipformer-large-streaming/encoder-chunk-32-left-128.onnx \
        --decoder zipformer-large-streaming/decoder-chunk-32-left-128.onnx \
        --joiner zipformer-large-streaming/joiner-chunk-32-left-128.onnx \
        --streaming 1 \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx fp16 model
    zipformer inference \
        --encoder zipformer-large-streaming/encoder-chunk-32-left-128.fp16.onnx \
        --decoder zipformer-large-streaming/decoder-chunk-32-left-128.onnx \
        --joiner zipformer-large-streaming/joiner-chunk-32-left-128.fp16.onnx \
        --streaming 1 \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav

    # onnx int8 model
    zipformer inference \
        --encoder zipformer-large-streaming/encoder-chunk-32-left-128.int8.onnx \
        --decoder zipformer-large-streaming/decoder-chunk-32-left-128.onnx \
        --joiner zipformer-large-streaming/joiner-chunk-32-left-128.int8.onnx \
        --streaming 1 \
        --model-type onnx \
        --tokens data/tokens.txt \
        data/en.wav data/zh.wav
    ```

=== "Auto-downloading from HuggingFace"

    ```bash
    # jit script model
    zipformer inference \
        --hf-model pkufool/zipformer-large-streaming \
        --streaming 1 \
        --model-type jit \
        data/en.wav data/zh.wav

    # onnx model
    zipformer inference \
        --hf-model pkufool/zipformer-large-streaming \
        --streaming 1 \
        --model-type onnx \
        data/en.wav data/zh.wav

    # onnx fp16 model
    zipformer inference \
        --hf-model pkufool/zipformer-large-streaming \
        --streaming 1 \
        --dtype fp16 \
        --model-type onnx \
        data/en.wav data/zh.wav

    # onnx int8 model
    zipformer inference \
        --hf-model pkufool/zipformer-large-streaming \
        --streaming 1 \
        --dtype int8 \
        --model-type onnx \
        data/en.wav data/zh.wav
    ```

##### Python API

* CTC head inference

=== "Using downloaded models"

    ```python
    from zipformer import inference

    # jit script model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="jit",
        model="zipformer-large-streaming/jit_model-chunk-32-left-128.pt",
        tokens="data/tokens.txt",
        streaming=True,
        ctc=True,
    )

    # onnx model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        model="zipformer-large-streaming/ctc-chunk-32-left-128.onnx",
        tokens="data/tokens.txt",
        streaming=True,
        ctc=True,
    )

    # onnx fp16 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        model="zipformer-large-streaming/ctc-chunk-32-left-128.fp16.onnx",
        tokens="data/tokens.txt",
        streaming=True,
        ctc=True,
    )

    # onnx int8 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        model="zipformer-large-streaming/ctc-chunk-32-left-128.int8.onnx",
        tokens="data/tokens.txt",
        streaming=True,
        ctc=True,
    )
    ```

=== "Auto-downloading from HuggingFace"

    ```python
    from zipformer import inference

    # jit script model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="jit",
        hf_model="pkufool/zipformer-large-streaming",
        streaming=True,
        ctc=True,
    )

    # onnx model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large-streaming",
        streaming=True,
        ctc=True,
    )

    # onnx fp16 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large-streaming",
        streaming=True,
        ctc=True,
        dtype="fp16",
    )

    # onnx int8 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large-streaming",
        streaming=True,
        ctc=True,
        dtype="int8",
    )
    ```

* Transducer head inference

=== "Using downloaded models"

    ```python
    from zipformer import inference

    # jit script model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="jit",
        model="zipformer-large-streaming/jit_model-chunk-32-left-128.pt",
        tokens="data/tokens.txt",
        streaming=True,
    )

    # onnx model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        encoder="zipformer-large-streaming/encoder-chunk-32-left-128.onnx",
        decoder="zipformer-large-streaming/decoder-chunk-32-left-128.onnx",
        joiner="zipformer-large-streaming/joiner-chunk-32-left-128.onnx",
        tokens="data/tokens.txt",
        streaming=True,
    )

    # onnx fp16 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        encoder="zipformer-large-streaming/encoder-chunk-32-left-128.fp16.onnx",
        decoder="zipformer-large-streaming/decoder-chunk-32-left-128.onnx",
        joiner="zipformer-large-streaming/joiner-chunk-32-left-128.fp16.onnx",
        tokens="data/tokens.txt",
        streaming=True,
    )

    # onnx int8 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        encoder="zipformer-large-streaming/encoder-chunk-32-left-128.int8.onnx",
        decoder="zipformer-large-streaming/decoder-chunk-32-left-128.onnx",
        joiner="zipformer-large-streaming/joiner-chunk-32-left-128.int8.onnx",
        tokens="data/tokens.txt",
        streaming=True,
    )
    ```

=== "Auto-downloading from HuggingFace"

    ```python
    from zipformer import inference

    # jit script model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="jit",
        hf_model="pkufool/zipformer-large-streaming",
        streaming=True,
    )

    # onnx model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large-streaming",
        streaming=True,
    )

    # onnx fp16 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large-streaming",
        streaming=True,
        dtype="fp16",
    )

    # onnx int8 model
    results = inference(
        ["data/en.wav", "data/zh.wav"],
        model_type="onnx",
        hf_model="pkufool/zipformer-large-streaming",
        streaming=True,
        dtype="int8",
    )
    ```
