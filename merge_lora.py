from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import argparse
import yaml

### argparseを設定
parser = argparse.ArgumentParser(description="トレーニング設定")
parser.add_argument('--config', type=str, required=True, help='YAML設定ファイルのパス')
args = parser.parse_args()

### 設定用YAMLファイルの読み込み
with open(args.config, 'r',encoding='utf-8') as file:
    config = yaml.safe_load(file)

    file_config = config['file_config']
    prompt_config = config['prompt_config']
    tokenizer_config = config['tokenizer_config']

### model path
model_path = file_config['model_path'] ### base model
output_path = file_config['output_dir']+"/"+file_config['title']


# モデルの読み込み
model = AutoPeftModelForCausalLM.from_pretrained(
    output_path, 
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
