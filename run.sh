# MP=8
# TARGET_FOLDER=4999_llama
# torchrun --nproc_per_node $MP example.py --ckpt_dir $TARGET_FOLDER --tokenizer_path $TARGET_FOLDER/$TARGET_FOLDER/llamav4.model

# MP=1
# TARGET_FOLDER=LLaMA
# torchrun --nproc_per_node $MP example.py --ckpt_dir $TARGET_FOLDER/7B --tokenizer_path $TARGET_FOLDER/tokenizer.model

# # python example.py --ckpt_dir $TARGET_FOLDER/7B --tokenizer_path $TARGET_FOLDER/tokenizer.model



MP=1
TARGET_FOLDER=LLaMA
torchrun --nproc_per_node $MP \
    example.py \
    --ckpt_dir $TARGET_FOLDER/7B \
    --tokenizer_path $TARGET_FOLDER/tokenizer.model \
    --temperature 0
