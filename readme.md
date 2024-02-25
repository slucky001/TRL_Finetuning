# TRLを用いたSFT Finetuning Sampleコード
## 概要
Npaka先生のコードを元にローカルでのFinetuning用にカスタマイズしたものです。<br>
LoRA作成～MergeModel作成、MergeModelの評価までをこのリポジトリで管理します。
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



Windowsの場合、bitsandbytesの入れ替えが必要です。<br>
2023.11.13現在、Latest Installすると 0.41.2 post2がInstallされますがcuda11.8と相性が悪いようです。<br>

```
python -m pip install bitsandbytes==0.41.1 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```
### 使用方法
#### trl sft training
以下のコマンドでtrainsettings内のyamlファイルに記述した設定に従ってLoRAアダプタを作成してくれます。<br>
デフォルトではcalm2-7b-chat Learning rate 2e-4,step300の設定を入れておきます。<br>
LoRAConig,TrainingArgumentsは引数をほぼそのままYAMLにしてあるので追加、修正そのまま可能ですが型に注意が必要です。<br>
※torch.float16など

```
trl_sft_training.py --config .\trainsettings\train_config.yaml
```
以下のコマンドでサンプルデータの"cyberagent/calm2-7b-chat"での確認を行えます。
```
generate_lora.py
```

tensorboardのreportを出力するので、以下のコマンドで学習結果を確認できます。<br>
(出力フォルダをデフォルトから変更していない場合)
```
tensorboard --logdir ./output
```

#### Test output

generate_config内のgenerate_config.yamlを読み込むことで指定したベースモデルでLoRAを使用した場合のテスト出力できます。
generate_config.yaml内のfile_configは、training用のconfig.yamlとあわせてください。
```
generate_lora.py --config .\generate_config\generate_config.yaml
```
#### Merge Model
generate_config内のgenerate_config.yamlを読み込むことでテスト出力できます。
file_configは、training用のconfig.yamlを指定してください。

```
merge_lora.py --config .\trainingsettings\training_config.yaml
```

### Change log
- 2023.11.20
    - T5Tokenizer用のsentencepieceをrequired.txtに追加(LINE1.7b用)
    - LINE1.7b sft用のサンプル設定を追加
    - YAMLにタイトル欄を設けて、アダプタはoutputの下にタイトル名のフォルダに出力するよう変更
    ※たくさん出力して評価するため。
    - generate_lora用にgenerate_config.yamlを作成(lineのみすり合わせ済み)
    - merge_lora.pyを設定ファイル化。設定はトレーニングと共用。
-2023.11.21
    - generate_ckpt.pyを一部設定ファイル化。
-2023.11.25
    - 評価用のmerge+generateを作成。
        training configから、各チェックポイントでマージを行い、評価用入力からの評価用の出力を行う。
        …にあたり、とりあえず、チェックポイント全マージの前半分作成
        …残り後半test.csvからinput.csvを読み込んで
-2024.02.25
    - ターゲットモジュールはyamlのpeft_configから切り離した。
        記述がなければLM_HEAD以外の全てのLinerLayerを取得してターゲットモジュールにする
        記述があれば書いたレイヤをターゲットモジュールにする
---
### Todo
- [x] lora test用のgenerate_lora設定ファイル化(まずLINE1.7b)
- [x] lora mergeの設定ファイル化(まずLINE1.7b)
- [x] merge model用のgenerateを設定ファイル化(とりあえずTokenizer,special_token_idあたり)
- [ ] merge model用のgenerateの入出力をファイルにする
- [ ] 上記、設定ファイルのcalm2-7b-chat用作成と調整
- [ ] merge後の出力評価用スクリプト対応
- [ ] merge modelのgguf化 , ctranslate2対応
- [ ] merge modelテスト用のsimple webui対応
