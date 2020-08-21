#!/usr/bin/env pythnon
import h5py
import torch
#from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
from bisect import bisect_right
from glob import glob
import numpy as np
import math

class HEPGNNDataset(PyGDataset):
    def __init__(self, **kwargs):
        super(HEPGNNDataset, self).__init__(None, transform=None, pre_transform=None)
        self.isLoaded = False
        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "weight", "label", "fileIdx"])

    def len(self):
        return int(self.maxEventsList[-1])

    def get(self, idx):
        if not self.isLoaded: self.initialize()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = idx-int(offset)

        label = self.labelList[fileIdx][idx]
        weight = self.weightList[fileIdx][idx]
        rescale = self.rescaleList[fileIdx][idx]
        etas = self.etaList[fileIdx][idx]
        phis = self.phiList[fileIdx][idx]
        nodes1 = self.nodes1List[fileIdx][idx]
        nodes2 = self.nodes2List[fileIdx][idx]

        feats = torch.Tensor(np.stack([x[idx] for x in self.featsList[fileIdx]]).T)
        poses = torch.Tensor(np.stack([etas, phis]).T)
        graph = torch.tensor(np.stack([nodes1, nodes2]).astype(dtype=np.dtype('int64')))

        data = PyGData(x = feats, pos = poses, edge_index = graph, y=label,
                       weight=weight, rescale=rescale)
        return data

    def addSample(self, procName, fNamePattern, weight=1, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        print(procName, fNamePattern)

        for fName in glob(fNamePattern):
            if not fName.endswith(".h5"): continue
            fileIdx = len(self.fNames)
            self.fNames.append(fName)

            info = {
                'procName':procName, 'weight':weight, 'nEvent':0,
                'label':0, ## default label, to be filled later
                'fileName':fName, 'fileIdx':fileIdx,
            }
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)

    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label
      
    def initialize(self):
        if self.isLoaded: return

        print(self.sampleInfo)

        self.labelList = []
        self.weightList = []
        self.rescaleList = []
        self.nodes1List = []
        self.nodes2List = []
        self.etaList = []
        self.phiList = []
        self.featsList = []

        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
            print("Loading files... (%d/%d) %s" % (i+1,nFiles,fName), end='\r')

            data = h5py.File(fName, 'r', libver='latest', swmr=True)
            nEvent = len(data['events/weights'])
            self.sampleInfo.loc[i, 'nEvent'] = nEvent

            ## set label and weight
            label = self.sampleInfo['label'][i]
            labels = torch.ones(nEvent, dtype=torch.int32, requires_grad=False)*label
            self.labelList.append(labels)
            weight = self.sampleInfo['weight'][i]
            weights = torch.ones(nEvent, dtype=torch.float32, requires_grad=False)*weight
            self.weightList.append(weights)
            self.rescaleList.append(torch.ones(nEvent, dtype=torch.float32, requires_grad=False))

            ## Load particles
            jets_eta = data['jets/eta']
            jets_phi = data['jets/phi']
            varNames = [varName for varName in data['jets'].keys() if varName not in ("eta", "phi")]
            jets_feats = [data['jets/'+varName] for varName in varNames]

            self.etaList.append(jets_eta)
            self.phiList.append(jets_phi)
            self.featsList.append(jets_feats)

            ## Load graphs
            nodes1 = data['graphs/nodes1']
            nodes2 = data['graphs/nodes2']
            self.nodes1List.append(nodes1)
            self.nodes2List.append(nodes2)

        print("")

        ## Compute cumulative sums of nEvent, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent'])))

        ## Compute sum of weights for each label categories
        sumWByLabel = {}
        sumEByLabel = {}
        for label in self.sampleInfo['label']:
            label = int(label)
            w = self.sampleInfo[self.sampleInfo.label==label]['weight']
            e = self.sampleInfo[self.sampleInfo.label==label]['nEvent']
            sumWByLabel[label] = (w*e).sum()
            sumEByLabel[label] = e.sum()
        ## Find overall rescale for the data imbalancing problem - fit to the category with maximum entries
        maxSumELabel = max(sumEByLabel, key=lambda key: sumEByLabel[key])
        maxWMaxSumELabel = self.sampleInfo[self.sampleInfo.label==maxSumELabel]['weight'].max()
        #meanWMaxSumELabel = self.sampleInfo[self.sampleInfo.label==maxSumELabel]['weight'].mean()
        #print()
        #print('label with maxEvent=', maxSumELabel)
        #print('  with sumE=', sumEByLabel[maxSumELabel], ' sumW=', sumWByLabel[maxSumELabel])
        #print('  sf of this label=', sumEByLabel[maxSumELabel]/sumWByLabel[maxSumELabel])
        #print('  max weight of this label=', maxWMaxSumELabel*sumEByLabel[maxSumELabel]/sumWByLabel[maxSumELabel])
        #print('  mean weight of this label=', meanWMaxSumELabel*sumEByLabel[maxSumELabel]/sumWByLabel[maxSumELabel])

        ## Find rescale factors - make average weight to be 1 for each cat in the training step
        for fileIdx in self.sampleInfo['fileIdx']:
            label = self.sampleInfo.loc[self.sampleInfo.fileIdx==fileIdx, 'label']
            for l in label: ## this loop runs only once, by construction.
                self.rescaleList[fileIdx] *= (sumEByLabel[maxSumELabel]/sumWByLabel[l])
                #print("@@@ Scale sample label_%d(sumE=%g,sumW=%g)->label_%d, sf=%f" % (l, sumEByLabel[l], sumWByLabel[l], maxSumELabel, sf))
                break ## this loop runs only once, by construction. this break is just for a confirmation

        self.isLoaded = True
