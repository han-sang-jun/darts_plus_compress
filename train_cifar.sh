CUDA_VISIBLE_DEVICES=0 python train_cifar.py \
--tmp_data_dir '../../datas/cifar-10' \
--save './experiments/output01/' --auxiliary --cutout --batch_size 128 \
--layers 20 --init_channels 36 --arch geno01