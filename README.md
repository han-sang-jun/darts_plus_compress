# darts+ with compression
install pytorch with anaconda
```buildoutcfg
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```
search base architecture with cifar-10
```buildoutcfg
$ sh train_search.sh
```
compress the architecture
```buildoutcfg
$ sh compress.sh
```
evaluate the compressed architecture with cifar-10
```buildoutcfg
$ sh train_cifar.sh
```