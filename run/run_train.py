#!/usr/bin/env python
import numpy as np
import argparse
import sys, os
import subprocess
import csv, yaml
import math

import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
print("NTHREADS=", nthreads, "CPU_COUNT=", os.cpu_count())
torch.set_num_threads(nthreads)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', action='store', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('-o', '--outdir', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--lr', action='store', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batchPerStep', action='store', type=int, default=1, help='Number of batches per step (to emulate all-reduce)')
parser.add_argument('--shuffle', action='store', type=bool, default=True, help='Shuffle batches for each epochs')
parser.add_argument('--optimizer', action='store', choices=('sgd', 'adam', 'radam', 'ranger'), default='adam', help='optimizer to run')
parser.add_argument('--device', action='store', type=int, default=-1, help='device name')
parser.add_argument('-c', '--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')

args = parser.parse_args()
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

if args.device >= 0: torch.cuda.set_device(args.device)

if not os.path.exists(args.outdir): os.makedirs(args.outdir)
modelFile = os.path.join(args.outdir, 'model.pkl')
weightFile = os.path.join(args.outdir, 'weight.pkl')
predFile = os.path.join(args.outdir, 'predict.npy')
trainingFile = os.path.join(args.outdir, 'train.csv')
resourceByCPFile = os.path.join(args.outdir, 'resourceByCP.csv')
resourceByTimeFile = os.path.join(args.outdir, 'resourceByTime.csv')

proc = subprocess.Popen(['python', '../scripts/monitor_proc.py', '-t', '1',
                        '-o', resourceByTimeFile, '%d' % os.getpid()],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

import time
class TimeHistory():#tf.keras.callbacks.Callback):
    def on_train_begin(self):
        self.times = []
    def on_epoch_begin(self):
        self.epoch_time_start = time.time()
    def on_epoch_end(self):
        self.times.append(time.time() - self.epoch_time_start)

sys.path.append("../scripts")

sys.path.append("../python")
from HEPGNNDataset import HEPGNNDataset as MyDataset

myDataset = MyDataset()
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    myDataset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    myDataset.setProcessLabel(name, sampleInfo['label'])
myDataset.initialize()

lengths = [int(0.6*len(myDataset)), int(0.2*len(myDataset))]
lengths.append(len(myDataset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
trnDataset, valDataset, testDataset = torch.utils.data.random_split(myDataset, lengths)
torch.manual_seed(torch.initial_seed())

kwargs = {'num_workers':min(config['training']['nDataLoaders'], nthreads), 'pin_memory':False}
#kwargs = {'pin_memory':True}
#if torch.cuda.is_available():
#    kwargs['pin_memory'] = True

trnLoader = DataLoader(trnDataset, batch_size=args.batch, shuffle=args.shuffle, **kwargs)
valLoader = DataLoader(valDataset, batch_size=args.batch, shuffle=False, **kwargs)

## Build model
from ModelDefault import MyModel
model = MyModel()
torch.save(model, modelFile)
device = 'cpu'
if torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'

if args.optimizer == 'radam':
    from optimizers.RAdam import RAdam
    optm = RAdam(model.parameters(), lr=args.lr)
elif args.optimizer == 'ranger':
    from optimizers.RAdam import RAdam
    from optimizers.Lookahead import Lookahead
    optm_base = RAdam(model.parameters(), lr=args.lr)
    optm = Lookahead(optm_base)
elif args.optimizer == 'adam':
    optm = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optm = optim.SGD(model.parameters(), lr=args.lr)
else:
    print("Cannot find optimizer in the list")
    exit()

with open(args.outdir+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.write(str(model))
    fout.close()

from tqdm import tqdm
from sklearn.metrics import accuracy_score
bestModel, bestAcc = {}, -1
try:
    timeHistory = TimeHistory()
    timeHistory.on_train_begin()
    history = {'time':[], 'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}

    for epoch in range(args.epoch):
        timeHistory.on_epoch_begin()

        model.train()
        trn_loss, trn_acc = 0., 0.
        optm.zero_grad()
        for i, batch in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, args.epoch))):
            batch = batch.to(device)

            pos, feats = batch.pos, batch.x
            label = batch.y.float()
            weight = batch.weight*batch.rescale

            pred = model(batch)
            crit = torch.nn.BCEWithLogitsLoss(weight=weight)
            if device == 'cuda': crit = crit.cuda()
            l = crit(pred.view(-1), label)
            l.backward()
            if i % args.batchPerStep == 0 or i+1 == len(trnLoader):
                optm.step()
                optm.zero_grad()

            trn_loss += l.item()
            trn_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0), sample_weight=weight.to('cpu'))

        trn_loss /= len(trnLoader)
        trn_acc  /= len(trnLoader)

        model.eval()
        val_loss, val_acc = 0., 0.
        for i, batch in enumerate(tqdm(valLoader)):
            batch = batch.to(device)

            pos, feats = batch.pos, batch.x
            label = batch.y.float()
            weight = batch.weight*batch.rescale

            pred = model(batch)
            crit = torch.nn.BCEWithLogitsLoss(weight=weight)
            if device == 'cuda': crit = crit.cuda()
            l = crit(pred.view(-1), label)

            val_loss += l.item()
            val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0), sample_weight=weight.to('cpu'))
        val_loss /= len(valLoader)
        val_acc  /= len(valLoader)

        if bestAcc < val_acc:
            bestModel = model.state_dict()
            bestAcc = val_acc
            torch.save(bestModel, weightFile)

        timeHistory.on_epoch_end()
        history['loss'].append(trn_loss)
        history['acc'].append(trn_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        history['time'].append(timeHistory.times[-1])
        with open(trainingFile, 'w') as f:
            writer = csv.writer(f)
            keys = history.keys()
            writer.writerow(keys)
            for row in zip(*[history[key] for key in keys]):
                writer.writerow(row)

except KeyboardInterrupt:
    print("Training finished early")

