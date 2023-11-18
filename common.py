### prompt、dataset作成
import bitsandbytes as bnb

### prompt 生成
def generate_prompt(data_point, prefixes):
    user_part = f"{prefixes['user_prefix']}{data_point.get('instruction', '')}"
    assistant_part = f"{prefixes['assistant_prefix']}{data_point.get('output', '')}"
    input_part = f"{prefixes['input_prefix']}{data_point.get('input', '')}"
    system_part = f"{prefixes['system_prefix']}{data_point.get('system', '')}"
    
    result = f"{system_part}{user_part}{input_part}{assistant_part}"
    
    return result

### 学習用 dataset作成
def add_text(example, prefixes):
    example['text'] = generate_prompt(example, prefixes)
    
    # 必要なキー以外を削除(textで教師ナシ)
    keys_to_keep = ['text']
    keys_to_delete = [key for key in example if key not in keys_to_keep]
    for key in keys_to_delete:
        del example[key]

    return example

### Bitsandbytes NF4のLiner layerListを得る
def find_all_linear_names_4bit_bnb(model):
    cls = bnb.nn.Linear4bit  # (default:torch.nn.Linear,4bit:bnb.nn.Linear4bit,8bit:bnb.nn.Linear8bitLt)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
