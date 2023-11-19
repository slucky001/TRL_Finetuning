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
prompt = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\nUSER: お好み焼きの作り方を詳しく教えて。\nASSISTANT: "
prompt = "あなたは誠実で優秀な日本人のアシスタントです。\nUSER: お好み焼きの作り方を詳しく教えて。\nASSISTANT:"
prompt = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\nUSER: お好み焼きの作り方を詳しく教えて。\nASSISTANT:"

# 推論の実行
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
output_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=200,
    repetition_penalty = 1.3,
    do_sample=True,
    #temperature=0.3, ### 評価のためにtemperatureは極力下げる
    temperature=0.01,
    pad_token_id=tokenizer.pad_token_id, ### youri パッドないのでeosにしろ、ってTransformersから怒られるのでこうした
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
output = tokenizer.decode(output_ids.tolist()[0])
print(output)