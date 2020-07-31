#!/usr/bin/env pythnon
#import h5py
import torch
#from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
from bisect import bisect_right
from glob import glob
import numpy as np
import uproot
import math
import numba, awkward, awkward.numba

@numba.njit(nogil=True, fastmath=True)
def buildGraph(jetss_eta, jetss_phi):
    prange = numba.prange
    maxDR2 = 1.2*1.2 ## maximum deltaR value to connect two jets

    graphs = [[[0], [0]]]
    graphs.pop()
    nEvent = len(jetss_eta)
    for ievt in prange(nEvent):
        jets_eta = jetss_eta[ievt]
        jets_phi = jetss_phi[ievt]
        nJet = len(jets_eta)

        nodes1, nodes2 = [], []
        for i in prange(nJet):
            for j in prange(i):
                dEta = jets_eta[i]-jets_eta[j]
                dPhi = jets_phi[i]-jets_phi[j]
                ## Move dPhi to [-pi,pi] range
                if   dPhi >= math.pi: dPhi -= 2*math.pi
                elif dPhi < -math.pi: dPhi += 2*math.pi
                ## Compute deltaR^2 and ask it is inside of our ball
                dR2 = dEta*dEta + dPhi*dPhi
                if dR2 > maxDR2: continue
                nodes1.append(i)
                nodes2.append(j)
        graphs.append([nodes1+nodes2, nodes2+nodes1])

    return graphs

class DelphesDataset(PyGDataset):
    def __init__(self, **kwargs):
        super(DelphesDataset, self).__init__(None, transform=None, pre_transform=None)
        self.treeName = "Delphes"
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

        graph = self.graphList[fileIdx][idx]
        label = self.labelList[fileIdx][idx]
        weight = self.weightList[fileIdx][idx]
        rescale = self.rescaleList[fileIdx][idx]
        poses = self.posList[fileIdx][idx]
        feats = self.featList[fileIdx][idx]

        data = PyGData(x = feats, pos = poses, edge_index = graph, y=label,
                       weight=weight, rescale=rescale)
        return data

    def addSample(self, procName, fNamePattern, weight=1, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))

        for fName in glob(fNamePattern):
            if not fName.endswith(".root"): continue
            fileIdx = len(self.fNames)
            self.fNames.append(fName)

            info = {
                'procName':procName, 'weight':weight, 'nEvents':0,
                'label':0, ## default label, to be filled later
                'fileName':fName, 'fileIdx':fileIdx,
            }
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)

    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label
      
    def initialize(self):
        if self.isLoaded: return

        self.labelList = []
        self.weightList = []
        self.rescaleList = []
        self.graphList = []
        self.posList = []
        self.featList = []

        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
            f = uproot.open(fName)
            if self.treeName not in f: continue
            tree = f[self.treeName]
            if tree == None: continue

            print("Loading files... (%d/%d) %s" % (i+1,nFiles,fName), end='\r')

            nEvents = len(tree)
            self.sampleInfo.loc[i, 'nEvents'] = nEvents

            ## set label and weight
            label = self.sampleInfo['label'][i]
            labels = torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*label
            self.labelList.append(labels)
            weight = self.sampleInfo['weight'][i]
            weights = torch.ones(nEvents, dtype=torch.float32, requires_grad=False)*weight
            self.weightList.append(weights)
            self.rescaleList.append(torch.ones(nEvents, dtype=torch.float32, requires_grad=False))

            ## Load particles
            jetss_eta, jetss_phi = [tree["Jet"][x].array() for x in ("Jet.Eta", "Jet.Phi",)]
            jetss_feats = [tree["Jet"][x].array() for x in ("Jet.PT", "Jet.Mass", "Jet.BTag",)]

            poses, feats = [], []
            for iEvent in range(nEvents):
                posi, feati = [], []
                for i in range(len(jetss_eta[iEvent])):
                    posi.append([jetss_eta[iEvent][i], jetss_phi[iEvent][i]])
                    feati.append([x[iEvent][i] for x in jetss_feats])
                poses.append(torch.Tensor(posi))
                feats.append(torch.Tensor(feati))
            self.posList.append(poses)
            self.featList.append(feats)

            ## Build graphs
            g = buildGraph(jetss_eta, jetss_phi)
            g = [torch.LongTensor(x) for x in g]
            self.graphList.append(g)

        print("")

        ## Compute sum of weights for each label categories
        sumWByLabel = {}
        sumEByLabel = {}
        for label in self.sampleInfo['label']:
            label = int(label)
            sumWByLabel[label] = self.sampleInfo[self.sampleInfo.label==label]['weight'].sum()
            sumEByLabel[label] = self.sampleInfo[self.sampleInfo.label==label]['nEvents'].sum()
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

        ## Compute cumulative sums of nEvents, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvents'])))

        ## Find rescale factors - make average weight to be 1 for each cat in the training step
        for fileIdx in self.sampleInfo['fileIdx']:
            label = self.sampleInfo.loc[self.sampleInfo.fileIdx==fileIdx, 'label']
            for l in label: ## this loop runs only once, by construction.
                sf = sumWByLabel[maxSumELabel]/maxWMaxSumELabel/sumWByLabel[l]
                self.rescaleList[fileIdx] *= sf
                print("@@@ Scale sample label_%d(sumE=%g,sumW=%g)->label_%d, sf=%f" % (l, sumEByLabel[l], sumWByLabel[l], maxSumELabel, sf))

        #print(self.weightList)
        #print(self.rescaleList)

        self.isLoaded = True
