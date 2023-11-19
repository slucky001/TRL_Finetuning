from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    "./marged_model",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,  # 4bit量子化
    device_map={"": 0},
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(
    "./marged_model", 
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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