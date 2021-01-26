import argparse
from genotypes import Genotype

from train_search import model_search
from compress import model_compress
from train_cifar import train_model

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.02, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--eps_no_archs', type=int, default=15, help='epochs to do not train arch params')
parser.add_argument('--train_data_dir', type=str, help='train data dir')
parser.add_argument('--test_data_dir', type=str, help='test data dir')
parser.add_argument('--inter_nodes', type=int, default=4)
parser.add_argument('--stem_multiplier', type=int, default=3)
parser.add_argument('--stable_arch', type=int, default=5)
parser.add_argument('--residual_connection', type=bool, default=False)
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')

parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')

parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')

args = parser.parse_args()

result_genotypes = []

result_geno, best_arch_stable = model_search(args)
result_genotypes.append(str(result_geno))

args.epochs = 70
args.arch = str(result_geno)
args.layers = 11
result_geno, best_arch_stable = model_compress(args)
result_genotypes.append(str(result_geno))

args.arch = str(result_geno)
args.layers = 14
result_geno, best_arch_stable = model_compress(args)
result_genotypes.append(str(result_geno))

args.arch = str(result_geno)
args.layers = 17
result_geno, best_arch_stable = model_compress(args)
result_genotypes.append(str(result_geno))

print(result_genotypes)
args.tmp_data_dir = args.train_data_dir
args.epochs = 600
args.layers = 20
args.init_channels = 36
args.cutout = True
args.auxiliary = True
for geno in result_genotypes:
    args.arch = geno
    train_model(args)
    train_model(args)
    train_model(args)







