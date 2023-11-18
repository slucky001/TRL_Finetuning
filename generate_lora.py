### calm2 用 test generate ... 
### todo
### 設定と入力のYAML化

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# モデルの読み込み
model = AutoPeftModelForCausalLM.from_pretrained(
    "./output", 
    torch_dtype=torch.float16,
    load_in_4bit=True,  # 4bit量子化
    device_map="auto", 
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(
    "cyberagent/calm2-7b-chat", 
    trust_remote_code=True
)

# プロンプトの準備
prompt = "USER: 富士山の高さは？\nASSISTANT:"

# 推論の実行
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
output_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=100,
    repetition_penalty = 1.3,
    do_sample=True,
    temperature=0.3,
    eos_token_id=tokenizer.eos_token_id,
)
output = tokenizer.decode(output_ids.tolist()[0])
print(output)