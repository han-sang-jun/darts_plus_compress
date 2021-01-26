CUDA_VISIBLE_DEVICES=0 python compress.py \
--epochs 70 --eps_no_archs 15 \
--learning_rate 0.025 --arch_learning_rate 3e-4 --stable_arch 10 \
--layers 20 --init_channels 16 --stem_multiplier 3 --batch_size 128 --inter_nodes 4 \
--arch geno01_com --save './experiments/output01/' \
--train_data_dir '../../datas/cifar-10'

