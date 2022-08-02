# To specify where the models should be saved
# use DIFFUSION_BLOB_LOGDIR env variable
env DIFFUSION_BLOB_LOGDIR="./blobs" python3 image_train_v1.py --data_dir cifar_train
