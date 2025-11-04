# Commands to run: 
python src/prepare_data.py
python src/train_qlora.py
python src/infer.py --base_model Qwen/Qwen2.5-3B-Instruct --adapter_dir outputs\alex-qlora --tone direct --user "Give me 3 resume tips"
