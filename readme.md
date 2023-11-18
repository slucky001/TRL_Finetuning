# TRLを用いたSFT Finetuning Sampleコード
## 概要
Npaka先生のコードを元にローカルでのFinetuning用にカスタマイズしたものです。
- 各種設定をファイル化して引数化
- 汎用関数をcommon.pyに移動
- neftune noise alphaを勝手に追加

(参考:Google Colab + trl で SFT のQLoRAファインチューニングを試す)
https://note.com/npaka/n/na506c63b8cc9#558fa5ec-d62a-4e06-b2a8-bee9df08d859

## Install

- peftを使用する場合、torchバージョンに注意が必要です。
- neftune noiseを使用するため、transofomers 4.35.0以上が必要です。

```
pip install -r requirements.txt
```

- cuda11.8の場合
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

- cuda11.6の場合
```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```


Windowsの場合、bitsandbytesの入れ替えが必要です。
2023.11.13現在、Latest Installすると 0.41.2 post2がInstallされますがcuda11.8と相性が悪いようです。

```
python -m pip install bitsandbytes==0.41.1 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```
### 使用方法
以下のコマンドでtrainsettings内のyamlファイルに記述した設定に従ってLoRAアダプタを作成してくれます。
```
trl_sft_training.py --config .\trainsettings\train_config.yaml
```
tensorboardのreportを出力するので、以下のコマンドで学習結果を確認できます。
(出力フォルダをデフォルトから変更していない場合)
```
tensorboard --logdir ./output
```


---
### Todo
- [ ]  generate汎用化
- [ ]  merge対応
- [ ]  evaluation with gpt3.5turbo対応
- [ ]  gguf化 , ctranslate2対応
- [ ]  simple webui対応
