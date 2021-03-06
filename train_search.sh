CUDA_VISIBLE_DEVICES=0 python train_search.py --save './experiments/test/' \
--epochs 60 --eps_no_archs 15 \
--learning_rate 0.025 --arch_learning_rate 3e-4 --stable_arch 10 \
--layers 8 --init_channels 16 --stem_multiplier 3 --batch_size 96 --inter_nodes 4 \
--train_data_dir '../../datas/cifar-10'