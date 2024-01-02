### 各ステップのチェックポイントのadapterをマージして
### evaluationのtest.csvを読み込み
### responceにjsonファイルを出力

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import argparse
import yaml
import json
import os
import gc
import time
import sys
from datasets import load_dataset



### あるフォルダの配下のフォルダリストを得る
### チェックポイントのフォルダ指定用
def list_subfolders(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name)) and name.startswith("checkpoint")]


### argparseを設定
parser = argparse.ArgumentParser(description="トレーニング設定 生成設定")
parser.add_argument('--config', type=str, nargs='+', required=True, help='YAML設定ファイルのパス')
args = parser.parse_args()
configs =[]
### merge設定用YAMLファイルの読み込み
for config_path in args.config:
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        configs.append(config)

if len(configs) < 2:
    print("Error: At least two configuration files are required.--config training.yaml generate.yaml")
    sys.exit(1)

file_config = configs[0]['file_config']
tokenizer_config = configs[0]['tokenizer_config']

file_config = configs[1]['file_config']
prompt_config = configs[1]['prompt_config']
generate_config = configs[1]['generate_config']
modelload_config = configs[1]['modelload_config']


### model path (MergemodelのPathは固定。都度上書き。ストレージが大量に必要になるため。)
model_path = file_config['model_path'] ### base model
output_path = file_config['output_dir']+"/"+file_config['title'] ### LoraAdapterRoot

### prompt
system_prefix = prompt_config['system_prefix']
user_prefix = prompt_config['user_prefix']
assistant_prefix = prompt_config['assistant_prefix']
test_message = prompt_config['test_message']


subfolders = list_subfolders(output_path)
print("target folders:",subfolders)
print("generate_config",generate_config)
### 評価用input作成
ds = load_dataset('csv', data_files='./evaluation/test.csv')

first_input = ds['train'][0]['input']
print(first_input)
#exit()

if False:
    #### 前半：モデルマージ
    folder_path = os.path.join(output_path, folder)
    print("progressing folder...",folder_path)

    # モデルの読み込み
    model = AutoPeftModelForCausalLM.from_pretrained(
        folder_path, 
        torch_dtype=torch.float16,
        #load_in_4bit=True,  # 4bit量子化するとマージできない(らしい)
        device_map="auto", 
    )

    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        **tokenizer_config,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # マージして保存
    model = model.merge_and_unload()
    model.save_pretrained("./merged_model")
    tokenizer.save_pretrained('./merged_model')

    # いったんモデルを解放
    del model
    gc.collect()
    # CUDAキャッシュのクリア
    torch.cuda.empty_cache()

    time.sleep(10)

if True:
# ここで各inputに対して処理を行う
    #### 後半：出力
    ### モデルの準備
    model = AutoModelForCausalLM.from_pretrained(
        "./merged_model", ### 評価の過程でmerged_modelはテンポラリなのでフォルダ固定
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,  # 4bit量子化
        device_map={"": 0},
    )

    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(
        "./merged_model", 
        **tokenizer_config,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

### 一回生成だけで確認
inputs = [record['input'] for record in ds['train']]
results = []
### special token idの設定
generate_config['pad_token_id'] = getattr(tokenizer, generate_config['pad_token_id'])
generate_config['bos_token_id'] = getattr(tokenizer, generate_config['bos_token_id'])
generate_config['eos_token_id'] = getattr(tokenizer, generate_config['eos_token_id'])

for index, finput in enumerate(inputs):

    ### プロンプトの準備
    #prompt = f"{system_prefix}{user_prefix}{test_message}{assistant_prefix}"
    prompt = f"{system_prefix}{user_prefix}{finput}{assistant_prefix}"

    # 推論の実行
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')


    output_ids = model.generate(
        **input_ids.to(model.device),
        **generate_config,
    )
    output = tokenizer.decode(output_ids.tolist()[0])

    user_text_start = output.find(assistant_prefix)
    if user_text_start != -1:
        out_text = output[user_text_start + len(assistant_prefix):]
    else:
        out_text = ""

    ### jsonとしてアペンド
    result = {
        "index": index + 1,
        "input": finput,
        "output": out_text
    }
    results.append(result)

    print(output)

# JSONファイルに書き出し
with open("./evaluation/output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)






