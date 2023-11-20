from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# モデルの読み込み
model = AutoPeftModelForCausalLM.from_pretrained(
    "./output/line1.7b-sft_lr2e-4_do0.5_nef5_step300", 
    torch_dtype=torch.float16,
    #load_in_4bit=True,  # 4bit量子化するとマージできない(らしい)
    device_map="auto", 
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(
    "line-corporation/japanese-large-lm-1.7b-instruction-sft",
    use_fast=False,
    legacy=False,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# マージして保存
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
tokenizer.save_pretrained('./merged_model')
