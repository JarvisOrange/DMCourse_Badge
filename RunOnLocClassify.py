import numpy as np
import sys
import gc
import gzip
import pickle
#import openml
import os
import argparse
from dataset import get_dataset, get_handler
from model import get_net
import vgg
import resnet
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import time
import pdb
from scipy.stats import zscore
import pandas as pd

class logger():
    def __init__(self, path):
        self.path = path
        self.result = ""

    def log(self, result):
        self.result  += result +"\n"

    def save_result(self):
        with open(self.path, "w") as file:
            file.write(self.result)
        file.close()


from query_strategies import RandomSampling, BadgeSampling, \
                                BaselineSampling, LeastConfidence, MarginSampling, \
                                EntropySampling, CoreSet, ActiveLearningByLearning, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning, BaitSampling


parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
parser.add_argument('--nEnd', help = 'total number of points to query', type=int, default=5000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=128)
parser.add_argument('--rounds', help='number of rounds (0 does entire dataset)', type=int, default=0)
parser.add_argument('--trunc', help='dataset truncation (-1 is no truncation)', type=int, default=-1)
parser.add_argument('--aug', help='do augmentation (for cifar)', type=int, default=0)
parser.add_argument('--dummy', help='dummy input for indexing replicates', type=int, default=1)
opts = parser.parse_args()
print(opts, flush=True)

# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB)/ opts.nQuery)
DATA_NAME = opts.data

# regularization settings for bait
opts.lamb = 1
if 'CIFAR' in opts.data: opts.lamb = 1e-2



# load openml dataset if did is supplied

data_path = "./data/bj_dataset/poi_func_label_640000.npy"
embedding_path = "./data/bj_dataset/embedding_HIER_bj_dataset_128.npy"
embed = np.load(embedding_path)
data = np.load(data_path)
X = embed
y = data[:,1]
y = LabelEncoder().fit(y).transform(y)
opts.nClasses = int(max(y) + 1)
nSamps, opts.dim = np.shape(X)
testSplit = 0.1
inds = np.random.permutation(nSamps)
X = X[inds]
y = y[inds]


split =int((1. - testSplit) * nSamps)
while True:
    inds = np.random.permutation(split)
    if len(inds) > 50000:
        inds = inds[:50000]
    X_tr = X[:split]
    X_tr = X_tr[inds]
    X_tr = torch.Tensor(X_tr)

    y_tr = y[:split]
    y_tr = y_tr[inds]
    Y_tr = torch.Tensor(y_tr).long()

    X_te = torch.Tensor(X[split:])
    Y_te = torch.Tensor(y[split:]).long()

    if len(np.unique(Y_tr)) == opts.nClasses:
        break


args = {'transform':transforms.Compose([transforms.ToTensor()]),
        'n_epoch':10,
        'loader_tr_args':{'batch_size': 128, 'num_workers': 0},
        'loader_te_args':{'batch_size': 1000, 'num_workers': 0},
        'optimizer_args':{'lr': 0.01, 'momentum': 0},
        'transformTest':transforms.Compose([transforms.ToTensor()])}
handler = get_handler('other')




args['lr'] = opts.lr
args['modelType'] = opts.model
args['lamb'] = opts.lamb

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True


# mlp model class
class mlpMod(nn.Module):
    def __init__(self, dim, embSize=128, useNonLin=True):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.linear = nn.Linear(embSize, opts.nClasses, bias=False)
        self.useNonLin = useNonLin
    def forward(self, x):
        x = x.view(-1, self.dim)
        if self.useNonLin: emb = F.relu(self.lm1(x))
        else: emb = self.lm1(x)
        out = self.linear(emb)
        return out, emb
    def get_embedding_dim(self):
        return self.embSize

# load specified network

net = mlpMod(opts.dim, embSize=opts.nEmb)


if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()


# print info


if type(X_te) == torch.Tensor: X_te = X_te.numpy()
if opts.alg == 'rand': # random sampling
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'bait': # bait sampling
    strategy = BaitSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'conf': # confidence-based sampling
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'marg': # margin-based sampling
    strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'badge': # batch active learning by diverse gradient embeddings
    strategy = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'coreset': # coreset sampling
    strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'entropy': # entropy-based sampling
    strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'baseline': # badge but with k-DPP sampling instead of k-means++
    strategy = BaselineSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'albl': # active learning by learning
    albl_list = [LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args),
        CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)]
    strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
else:
    print('choose a valid acquisition function', flush=True)
    raise ValueError
print(DATA_NAME, flush=True)
print(type(strategy).__name__, flush=True)

Logger = logger("./result/"+opts.alg+'.txt')

# round 0 accuracy
strategy.train(verbose=False)
P, Preds = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)

def top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def cal_metric(pre_dists, labels, top_n_list=[1,2,3,4,5]):
    top_n_acc = [top_n_accuracy(labels, pre_dists, n) for n in top_n_list]
    score_series = pd.Series(top_n_acc,
                             index=['acc@{}'.format(n) for n in top_n_list])
    return score_series
res = cal_metric(Preds, Y_te)
Logger.log('Round 0: \ttesting acc@1 {}, acc@5 {}'.format(res['acc@1'], res['acc@5']))
print(str(opts.nStart) + '\ttesting accuracy {}'.format(acc[0]), flush=True)


for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd), flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # query
    output = strategy.query(NUM_QUERY)
    q_idxs = output
    idxs_lb[q_idxs] = True

    # update
    strategy.update(idxs_lb )
    strategy.train(verbose=False)

    # round accuracy
    P, Preds = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    res = cal_metric(Preds,Y_te)
    Logger.log('Round {}: \ttesting acc@1 {}, acc@5 {}'.format(rd, res['acc@1'], res['acc@5']))
    print(str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(acc[rd]), flush=True)
    if sum(~strategy.idxs_lb) < opts.nQuery: break
    if opts.rounds > 0 and rd == (opts.rounds - 1): break

Logger.save_result()