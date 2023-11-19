from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# モデルの読み込み
model = AutoPeftModelForCausalLM.from_pretrained(
    "./output", 
    torch_dtype=torch.float16,
    #load_in_4bit=True,  # 4bit量子化
    device_map="auto", 
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(
    "cyberagent/calm2-7b-chat",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# マージして保存
model = model.merge_and_unload()
#model.save_pretrained("./marged_model", safe_serialization=True)
model.save_pretrained("./marged_model")
tokenizer.save_pretrained('./marged_model')
