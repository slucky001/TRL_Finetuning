### generate with mergemodel
import argparse
import yaml
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

### argparseを設定
parser = argparse.ArgumentParser(description="生成設定")
parser.add_argument('--config', type=str, required=True, help='YAML設定ファイルのパス')
args = parser.parse_args()

### 設定用YAMLファイルの読み込み
with open(args.config, 'r',encoding='utf-8') as file:
    config = yaml.safe_load(file)

    file_config = config['file_config']
    prompt_config = config['prompt_config']
    tokenizer_config = config['tokenizer_config']
    generate_config = config['generate_config']
    modelload_config = config['modelload_config']

### model path
model_path = file_config['model_path'] ### base model
output_path = file_config['output_dir']+"/"+file_config['title']

### prompt
system_prefix = prompt_config['system_prefix']
user_prefix = prompt_config['user_prefix']
assistant_prefix = prompt_config['assistant_prefix']
test_message = prompt_config['test_message']


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

### プロンプトの準備
prompt = f"{system_prefix}{user_prefix}{test_message}{assistant_prefix}"

# 推論の実行
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')

### special token idの設定
generate_config['pad_token_id'] = getattr(tokenizer, generate_config['pad_token_id'])
generate_config['bos_token_id'] = getattr(tokenizer, generate_config['bos_token_id'])
generate_config['eos_token_id'] = getattr(tokenizer, generate_config['eos_token_id'])

output_ids = model.generate(
    **input_ids.to(model.device),
    **generate_config,
)
output = tokenizer.decode(output_ids.tolist()[0])
print(output)