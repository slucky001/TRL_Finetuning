### make LoRA with TRL SFTTrainer
import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

import common

### argparseを設定
parser = argparse.ArgumentParser(description="トレーニング設定")
parser.add_argument('--config', type=str, required=True, help='YAML設定ファイルのパス')
args = parser.parse_args()

### 設定用YAMLファイルの読み込み
with open(args.config, 'r',encoding='utf-8') as file:
    config = yaml.safe_load(file)

    file_config = config['file_config']
    prompt_config = config['prompt_config']
    bnb_config = config['bnb_4bit_config']
    tokenizer_config = config['tokenizer_config']
    peft_config = config['peft_config']
    training_config = config['training_config']
    other_config = config['other_config']

model_path = file_config['model_path']
dataset_path = file_config['dataset_path']
output_path = file_config['output_dir']+"/"+file_config['title']
max_seq_len = other_config['max_seq_length']

print("model_path:",model_path)
print("dataset_path:",dataset_path)
print("output_path:",output_path)
print("max_seq_length:",max_seq_len)

### dataset作成用ラッパー関数
def add_text_wrapper(example):
    return common.add_text(example, prompt_config)

print("reading dataset...")
dataset = load_dataset(dataset_path,split="train")

dataset = dataset.map(add_text_wrapper)

print(dataset)
print(dataset[0]["text"])

### bitsandbytes dtype変換 
print("bnb_4bit_compute_dtype:", bnb_config['bnb_4bit_compute_dtype'])
bnb_config['bnb_4bit_compute_dtype'] = getattr(torch, bnb_config['bnb_4bit_compute_dtype'])

### 量子化パラメータ
bnb_config = BitsAndBytesConfig(**bnb_config)

### モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    model_path,  # モデル名
    quantization_config=bnb_config,  # 量子化パラメータ
    device_map={"": 0}  # モデル全体をGPU0にロード
)

model.config.use_cache = False  # キャッシュ (学習時はFalse)
model.config.pretraining_tp = 1  # 事前学習で使用したテンソル並列ランク

### トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right" # fp16でのオーバーフロー問題対策

### ターゲットモジュールの取得
target_modules = common.find_all_linear_names_4bit_bnb(model)
print("target modules:",target_modules)


### LoRAパラメータ
peft_config = LoraConfig(target_modules=target_modules,**peft_config)

### 学習パラメータ
training_arguments = TrainingArguments(output_dir=output_path,**training_config)

### SFTパラメータ
trainer = SFTTrainer(
    model=model,  # モデル
    tokenizer=tokenizer,  # トークナイザー
    train_dataset=dataset,  # データセット
    dataset_text_field="text",  # データセットのtext列
    peft_config=peft_config,  # PEFTパラメータ
    args=training_arguments,  # 学習パラメータ
    #max_seq_length=None,  # 使用する最大シーケンス長
    max_seq_length=max_seq_len,  # 使用する最大シーケンス長
    packing=False,  # 同じ入力シーケンスに複数サンプルをパッキング(効率を高める)
)

### モデルの学習と保存
trainer.train()
trainer.model.save_pretrained(output_path)
