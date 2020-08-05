#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np
# import math
import torch
# from torch.utils.data import Dataset
import uproot
# from glob import glob
# from tqdm import tqdm
# from bisect import bisect_right
import h5py
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import sys, os
import subprocess
import csv
import math
from tqdm import tqdm
import torch.optim as optim
# from torch.utils.data import DataLoader


# %%
trainingFile = 'loss.csv'

# %%


def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class PointConvNet(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet, self).__init__()
        
        self.conv = PyG.PointConv(net)
        
    def forward(self, data, batch=None):
        x, pos, batch, edge_index = data.x, data.pos, data.batch, data.edge_index

#         print(x.shape)
#         print(pos.shape)
#         print(edge_index.shape)
#         print(edge_index)
        x = self.conv(x, pos, edge_index)
        return x, pos, batch

class PoolingNet(nn.Module):
    def __init__(self, net):
        super(PoolingNet, self).__init__()
        self.net = net

    def forward(self, x, pos, batch):
        x = self.net(torch.cat([x, pos], dim=1))
        x = PyG.global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.nChannel = 3
        
      
        self.conv1 = PointConvNet(MLP([self.nChannel+3, 64, 128]))
        self.pool = PoolingNet(MLP([128+3, 128]))

        self.fc = nn.Sequential(
            nn.Linear( 128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear( 64,   1),
        )
        
    def forward(self, data):
#         x, pos, batch, edge_index = data.x, data.pos, data.batch, data.edge_index

        x, pos, batch = self.conv1(data)
        x, pos, batch = self.pool(x, pos, batch)
        out = self.fc(x)
        return out


# %%
from bisect import bisect_right
from glob import glob
from torch.utils.data import Dataset
# from torch_geometric.data import Dataset
class HEPGNNDataset(Dataset):
    def __init__(self, **kwargs):
#     def __init__(self, root=None, **kwargs):
        super(HEPGNNDataset, self).__init__()
        self.isLoaded = False
        self.procFiles = {}
        self.procLabels = {}

        self.maxEventsList = [0,]
        self.graphList = []
        self.posList = []
        self.feaList = []
        self.edgeList = []
        self.labelsList = []
        self.weightsList = []
        self.rescaleList = []
        

    def __getitem__(self, idx):
        if not self.isLoaded: self.load()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = idx - offset

        
        
        feats = torch.Tensor(self.feaList[fileIdx][idx].reshape(-1,3))
        poses = torch.Tensor(self.posList[fileIdx][idx].reshape(-1,3))
        edges = torch.Tensor(self.edgeList[fileIdx][idx].reshape(2,-1))
        edges = edges.type(dtype = torch.long)
        label = self.labelsList[fileIdx][idx]
        weight = self.weightsList[fileIdx][idx]
        rescale = self.rescaleList[fileIdx][idx]
        

#         label = self.labels[index]
        
        data = Data(x = feats, pos = poses, edge_index = edges, y = label.item())
        data.ww = weight.item()*rescale.item()

        return data

    def __len__(self):
        return self.maxEventsList[-1]

    def addSample(self, procName, fileNames, weight=None, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fileNames))
        weightValue = weight ## Rename it just to be clear in the codes

        if procName not in self.procFiles:
            self.procFiles[procName] = [] ## this list keeps image index - later we will use this info to get total event and update weights, etc

        print(fileNames)
        for fileName in glob(fileNames):
#             print(fileName)

            f = h5py.File(fileName,'r', libver = 'latest', swmr=True)
            
            


            nEventsInFile = len(f['group']['pos'].get('pos'))
            self.maxEventsList.append(self.maxEventsList[-1]+nEventsInFile)
            
            weights = torch.ones(nEventsInFile, dtype=torch.float32, requires_grad=False)*weightValue
            
            labels  = torch.zeros(nEventsInFile, dtype=torch.float32, requires_grad=False) ## Put dummy labels, to set later by calling setProcessLabel()
            ## We will do this step for images later

            fileIdx = len(self.graphList)
            self.procFiles[procName].append(fileIdx)
            self.graphList.append(fileName)
            self.labelsList.append(labels)
            self.weightsList.append(weights)
            self.rescaleList.append(torch.ones(nEventsInFile, dtype=torch.float32, requires_grad=False))

    def setProcessLabel(self, procName, label):
        for i in self.procFiles[procName]:
            size = self.labelsList[i].shape[0]
            self.labelsList[i] = torch.ones(size, dtype=torch.float32, requires_grad=False)*label
            self.procLabels[procName] = label

    def initialize(self, logger=None):
        if logger: logger.update(annotation='Reweights by category imbalance')
        ## Compute sum of weights for each label categories
        sumWByLabel = {}
        sumEByLabel = {}
        for procName, fileIdxs in self.procFiles.items():
            label = self.procLabels[procName]
            if label not in sumWByLabel: sumWByLabel[label] = 0.
            if label not in sumEByLabel: sumEByLabel[label] = 0.
            procSumW = sum([sum(self.weightsList[i]) for i in fileIdxs])
            procSumE = sum([len(self.weightsList[i]) for i in fileIdxs])
            print("@@@ Process=%s nEvent=%d sumW=%.3f events/fb-1" % (procName, procSumE, procSumW.item()))
            sumWByLabel[label] += procSumW
            sumEByLabel[label] += procSumE

        ## Find rescale factors - make average weight to be 1 for each cat in the training step
        for procName, fileIdxs in self.procFiles.items():
            label = self.procLabels[procName]
            for i in fileIdxs: self.rescaleList[i] *= sumEByLabel[label]/sumWByLabel[label]

        ## Find overall rescale for the data imbalancing problem - fit to the category with maximum entries
        maxSumELabel = max(sumEByLabel, key=lambda key: sumEByLabel[key])
        for procName, fileIdxs in self.procFiles.items():
            label = self.procLabels[procName]
            if label == maxSumELabel: continue
            sf = sumEByLabel[maxSumELabel]/sumEByLabel[label]
            print("@@@ Scale up the sample", label, "->", maxSumELabel, sf)
            for i in fileIdxs: self.rescaleList[i] *= sf



    def load(self):
        if self.isLoaded: return
        for i, fName in enumerate(self.graphList):
            self.posList.append(h5py.File(fName, 'r', libver='latest', swmr=True)['group']['pos'].get('pos'))
            self.feaList.append(h5py.File(fName, 'r', libver='latest', swmr=True)['group']['fea'].get('fea'))
            self.edgeList.append(h5py.File(fName, 'r', libver='latest', swmr=True)['group']['edge'].get('edge'))
        self.isLoaded = True



# %%


myDataset = HEPGNNDataset()

myDataset.addSample("RPV", "/users/yewzzang/HEPGCN/RPV/*.hdf5", weight=1.0)
myDataset.addSample("QCD", "/users/yewzzang/HEPGCN/QCD/*.hdf5", weight=1.0)

myDataset.setProcessLabel("RPV", 1)
myDataset.setProcessLabel("QCD", 0) ## This is not necessary because the default is 0

lengths = [int(0.6*len(myDataset)), int(0.2*len(myDataset))]
lengths.append(len(myDataset)-sum(lengths))
torch.manual_seed(123456)
trnDataset, valDataset, testDataset = torch.utils.data.random_split(myDataset, lengths)
torch.manual_seed(torch.initial_seed())
print(lengths)


# %%


nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
print("NTHREADS=", nthreads, "CPU_COUNT=", os.cpu_count())
torch.set_num_threads(nthreads)


# %%


kwargs = {'num_workers':min(32, nthreads), 'pin_memory':False}
loader = DataLoader(trnDataset, batch_size = 512, shuffle = False, **kwargs)
valLoader = DataLoader(valDataset, batch_size = 512, shuffle = False, **kwargs)


# %%


model = MyModel()
device = 'cpu'
if torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'
optm = optim.Adam(model.parameters(), lr=0.001)
batchPerStep = 1

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

from sklearn.metrics import accuracy_score
bestModel, bestAcc = {}, -1
try:
    
       
    history = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
    
    for epoch in range(100):
        model.train()
        trn_loss, trn_acc = 0., 0.
        optm.zero_grad()
        for i, data in enumerate(tqdm(loader, desc='epoch %d/%d' % (epoch+1, 50))):

            data = data.to(device)
            label = data.y.float().to(device)
            weight = data.ww.float().to(device)

            pred = model(data)
            crit = torch.nn.BCEWithLogitsLoss(weight=weight)
            l = crit(pred.view(-1), label)
            l.backward()
            if i % batchPerStep == 0 or i+1 == len(loader):
                optm.step()
                optm.zero_grad()

            trn_loss += l.item()
            trn_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0))


        trn_loss /= len(loader)
        trn_acc  /= len(loader)
        print(trn_loss)
        print(trn_acc)
        model.eval()
        val_loss, val_acc = 0., 0.
        for i, data in enumerate(tqdm(valLoader)):
            print(i, data)

            data = data.to(device)
            label = data.y.float().to(device)
            weight = data.ww.float().to(device)
            
            pred = model(data)
            crit = torch.nn.BCEWithLogitsLoss(weight=weight)
            loss = crit(pred.view(-1), label)

            val_loss += loss.item()
            val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0))
        val_loss /= len(valLoader)
        val_acc  /= len(valLoader)
        print(val_loss)
        print(val_acc)
        


        
        
        history['loss'].append(trn_loss)
        history['acc'].append(trn_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        
        
        with open(trainingFile, 'w') as f:
            writer = csv.writer(f)
            keys = history.keys()
            writer.writerow(keys)
            for row in zip(*[history[key] for key in keys]):
                writer.writerow(row)
except KeyboardInterrupt:
    print("Training finished early")




# %%





# %%





# %%




