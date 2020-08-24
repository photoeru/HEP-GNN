#!/usr/bin/env python
import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('--type', action='store', type=str, choices=('trackpt', 'trackcount'), default='trackcount', help='image type')
parser.add_argument('--device', action='store', type=int, default=-1, help='device name')
parser.add_argument('-c', '--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')

args = parser.parse_args()
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

xmaxs = [5e3, 5e3, 20]
nbinsx = [50, 50, 20]
units = [1e-3, 1e-3, 1]
if args.type == 'trackpt':
    xmaxs[2] = xmaxs[0]
    nbinsx[2] = nbinsx[0]
    units[2] = units[0]

import torch
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    if args.device >= 0: torch.cuda.set_device(args.device)

import sys, os
sys.path.append("../python")
from HEPGNNDataset import HEPGNNDataset as MyDataset

myDataset = MyDataset()
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    myDataset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    myDataset.setProcessLabel(name, sampleInfo['label'])
myDataset.initialize()

procNames = myDataset.sampleInfo['procName'].unique()

#from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
lengths = [int(0.6*len(myDataset)), int(0.2*len(myDataset))]
lengths.append(len(myDataset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
trnDataset, valDataset, testDataset = torch.utils.data.random_split(myDataset, lengths)
torch.manual_seed(torch.initial_seed())

kwargs = {'num_workers':config['training']['nDataLoaders']}
allLoader = DataLoader(myDataset, batch_size=args.batch, shuffle=False, **kwargs)
trnLoader = DataLoader(trnDataset, batch_size=args.batch, shuffle=False, **kwargs)
valLoader = DataLoader(valDataset, batch_size=args.batch, shuffle=False, **kwargs)
testLoader = DataLoader(testDataset, batch_size=args.batch, shuffle=False, **kwargs)

import numpy as np

bins = [None, None, None]
imgHist_val_sig = [np.zeros(nbinsx[i]) for i in range(3)]
imgHist_val_bkg = [np.zeros(nbinsx[i]) for i in range(3)]
imgSum_val_sig, imgSum_val_bkg = None, None
sumE_sig, sumR_sig, sumW_sig = 0., 0., 0.
sumE_bkg, sumR_bkg, sumW_bkg = 0., 0., 0.
sumEs, sumRs, sumWs = {}, {}, {}
for procName in procNames:
    sumEs[procName] = 0.
    sumRs[procName] = 0.
    sumWs[procName] = 0.

from tqdm import tqdm
for i, batch in enumerate(tqdm(allLoader)):
    batch = batch.to(device)
    pos, feats = batch.pos, batch.x
    label = batch.y
    ws = batch.weight*batch.rescale
    #ws = (weights*rescales).float()

    for procIdx, procName in enumerate(procNames):
        ww = ws[batch.procIdxs==procIdx]
        sumEs[procName] += len(ww)
        sumRs[procName] += ww.sum()
        sumWs[procName] += batch.weight[batch.procIdxs==procIdx].sum()

        ww_sig = ws[(batch.procIdxs==procIdx) & (label==1)]
        ww_bkg = ws[(batch.procIdxs==procIdx) & (label==0)]
        sumE_sig += len(ww_sig)
        sumE_bkg += len(ww_bkg)
        sumR_sig += ww_sig.sum()
        sumR_bkg += ww_bkg.sum()
        sumW_sig += batch.weight[(batch.procIdxs==procIdx) & (label==1)].sum()
        sumW_bkg += batch.weight[(batch.procIdxs==procIdx) & (label==0)].sum()

print("-"*80)
print("sumEvent : signal=%d bkg=%d" % (sumE_sig, sumE_bkg))
print("sumResWgt: signal=%g bkg=%g" % (sumR_sig, sumR_bkg))
print("sumWeight: signal=%g bkg=%g" % (sumW_sig, sumW_bkg))
for procName in procNames:
    print("proc=%s sumE=%d sumR=%g sumW=%g" % (procName, sumEs[procName], sumRs[procName], sumWs[procName]))
print("-"*80)
print("sum=", sum(sumEs.values()), sum(sumWs.values()).item())
print("="*80)
