import subprocess

base_model = ".\\hf_ckpt" #マージモデルの出力先を指定
base_model = "./hf_ckpt" #マージモデルの出力先を指定
#base_model = "cyberagent/open-calm-7b" #マージモデルの出力先を指定
#base_model = "rinna/bilingual-gpt-neox-4b"
#base_model = "rinna/bilingual-gpt-neox-4b-instruction-ppo"
#base_model = "stockmark/gpt-neox-japanese-1.4b"
#base_model = "stabilityai/japanese-stablelm-instruct-alpha-7b"
#base_model = "matsuo-lab/weblab-10b-instruction-sft"
#base_model = "line-corporation/japanese-large-lm-3.6b-instruction-sft" #マージモデルの出力先を指定
#base_model = "meta-llama/Llama-2-13b-hf"
#base_model = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
#base_model = "llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0"
#base_model = "line-corporation/japanese-large-lm-1.7b-instruction-sft"
#base_model = "cyberagent/calm2-7b-chat"
#base_model = "elyza/ELYZA-japanese-Llama-2-13b-instruct"
base_model = ".\\merged_model"
output_dir = ".\\CtModel"
quantization_type = "int8_bfloat16"
#quantization_type = "int8_float32"
#quantization_type = "int8_float16"
#quantization_type = "float32"
#quantization_type = "int8"
#quantization_type = "int16"
#quantization_type = "auto"
#quantization_type = "bfloat16" 
#quantization_type = "float16" 


command = f"ct2-transformers-converter --model {base_model} --quantization {quantization_type} --output_dir {output_dir} --force"
#command = f"ct2-transformers-converter --model {base_model} --output_dir {output_dir} --force"

subprocess.run(command, shell=True)