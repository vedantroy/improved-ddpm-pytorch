MODEL_FLAGS="--model_path ./blobs/model000200.pt --num_samples 8 --batch_size 8"
rm -rf images && mkdir images
python3 image_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS 
