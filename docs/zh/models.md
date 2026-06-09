
## 中英文模型

### 非流式模型

| 名称 | 参数量 | 下载地址 | aishell test 1 / 2 |  wenetspeech test-net/meetting | Common Voice zh | kespeech test | librispeech test-clean / other | gigaspeech test | Common voice en | tedium test |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| xlarge-ctc | 298M | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-xlarge-ctc)  | 1.61 / 2.7  | 5.35 / 6.39 | 8.26 | 5.74 | 3.51 / 7.78 | 14.53 | 28.57 | 15.07 |
| large-ctc | 174M  | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-large-ctc)   | 2.51 / 3.51 | 6.23 / 6.67 | 7.96 | 8.95 | 2.62 / 5.17 | 10.73 | 12.99 | 10.11 |
| medium-ctc | 87M  | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-medium-ctc)  | 3.08 / 3.98 | 7.08 / 7.62 | 9.2  | 11.23| 3.01 / 6.06 | 11.22 | 15.28 | 10.38 |
| small-ctc  | 46M  | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-small-ctc)   | 4.82 / 5.5  | 10.09 / 11.3| 12.76|16.07 | 5.12 / 10.67|22.27  | 23.7  | 11.04 |
| large-rnnt | 174M | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-large-rnnt)  | 2.42 / 3.55 | 6.7 / 7.81  | 7.92 | 8.88 | 2.27 / 4.64 | 10.08 | 11.27 | 9.82  |
| medium-rnnt| 87M  | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-medium-rnnt) | 2.67 / 3.67 | 6.79 / 7.33 | 8.97 | 10.67| 2.61 / 5.36 | 10.56 | 12.94 | 10.06 |
| small-rnnt | 46M  | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-small-rnnt)  | 3.92 / 4.74 | 9.09 / 10.57|11.86 | 14.84|3.78 / 8.65  | 16.1  | 18.21 | 6.79  |


### 流式模型

> --chunk-size 16 --left-context-frames 128

| 名称 | 参数量 | 下载地址 | aishell test 1 / 2 |  wenetspeech test-net/meetting | Common Voice zh | kespeech test | librispeech test-clean / other | gigaspeech test | Common voice en | tedium test |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| large-ctc-streaming   | 174M | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-large-ctc-streaming)   | 3.78 / 4.71 | 8.65 / 10.54 | 11.8 | 15.35 | 3.74 / 8.5 | 12.32 | 19.7 | 10.92 |
| medium-ctc-streaming  | 87M  | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-medium-ctc-streaming)  | 4.46 / 5.09 | 9.74 / 11.21 | 12.68| 11.26 | 4.28 / 9.4 | 12.96 | 21.77| 11.26 |
| small-ctc-streaming   | 46M  | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-small-ctc-streaming)   | 6.7 / 7.24  | 12.92 / 16.45| 17.18| 23.32 | 19.4 / 29.66| 26.18| 33.52| 17.67 |
| large-rnnt-streaming  | 174M | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-large-rnnt-streaming)  | 3.53 / 4.48 | 8.31 / 10.27 | 11.99| 14.83 | 3.26 / 7.51 | 11.77| 17.53| 10.82 |
| medium-rnnt-streaming | 87M  | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-medium-rnnt-streaming) | 3.9 / 4.79  | 9.05 / 10.82 | 12.41| 17.89 | 3.64 / 8.08 | 12.13| 18.97| 10.9  |
| small-rnnt-streaming  | 46M  | [ModelScope](https://www.modelscope.cn/models/pkufool/zipformer-small-rnnt-streaming)  | 5.69 / 6.26 | 12.06 / 16.13| 16.51| 22.29 | 8.15 / 16.91| 19.77| 28.54| 14.23 |
