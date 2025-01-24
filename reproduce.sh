

# Clean up the output folder
mkdir output
rm -rf output/*.png

# Concept Learning
export source=bench # bench or ring
accelerate launch learning.py default \
  --instance_data_dir=./source_images/${source}/imgs \
  --instance_att_mask_dir=./source_images/${source}/masks \
  --output_dir=weights/$source-clic \
  --instance_prompt="a $source with <x> style"  \
  --learning_rate=1e-5  \
  --max_train_steps=500 \
  --attention_lambda=25e-2 \
  --context_lambda=50e-2 \
  --local_lambda=25e-2 \
  --modifier_token="<x>"  \
  --instance_mode="style" \
  --noaug

# Clean up the output folder
mkdir output
rm -rf output/*

# Concept Transfer
export target=chair # use chair for bench source image, use ring for ring source image
for gseed in 1 2 3 4 5 ; do

    rm -rf input/*
    cp target_images/$target/* input

    edit_prompt="a $target with <x> style" # token_id=4, which is the index of <x>
    for t_start in 15 20 ; do # We generate with both t_start=15 and 20. The optimal value highly depends on the source->target pairs themselves. Higher t_start means keeping more information of the original image.

        python transfer.py default weights=$source-clic/500 prompt="$edit_prompt" token_id=4 \
        seed=$gseed \
        t_start=$t_start \
        att_opt=10 \
        do_blending=True \
        do_guidance=True \
        tag=$target-$gseed-$t_start

    done

done