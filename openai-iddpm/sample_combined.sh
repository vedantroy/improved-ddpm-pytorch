MODEL_FLAGS="--model_path ep10.checkpoint --num_samples 8 --batch_size 8"
rm -rf images_combined && mkdir images_combined
python3 image_sample_combined.py $MODEL_FLAGS $DIFFUSION_FLAGS 
