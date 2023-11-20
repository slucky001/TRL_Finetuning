### test generate ... あくまでLoRAが焼けているかのため単体テスト文のみ

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import argparse
import yaml

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

### モデルの読み込み
modelload_config['torch_dtype'] = getattr(torch, modelload_config['torch_dtype']) ### torch_dtype.float16 or 32
model = AutoPeftModelForCausalLM.from_pretrained(
    output_path, 
    **modelload_config,
)

### トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_path,**tokenizer_config)

### プロンプトの準備
prompt = f"{system_prefix}{user_prefix}{test_message}{assistant_prefix}"

### 推論の実行
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