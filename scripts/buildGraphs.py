#!/usr/bin/env python3
import argparse
import h5py
import math
import numpy as np
import sys
import uproot
from glob import glob
from math import ceil
import numba
import numpy, numba, awkward, awkward.numba

if sys.version_info[0] < 3: sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument('input', nargs='+', action='store', type=str, help='input file name')
parser.add_argument('-o', '--output', action='store', type=str, help='output directory name', required=True)
#parser.add_argument('--format', action='store', default='Delphes', choices=("Delphes", "NanoAOD"), help='name of the main tree')
parser.add_argument('--data', action='store_true', default=False, help='Flag to set real data')
parser.add_argument('-n', '--nevent', action='store', type=int, default=-1, help='number of events to preprocess')
parser.add_argument('-c', '--chunk', action='store', type=int, default=1024, help='chunk size')
#parser.add_argument('--compress', action='store', choices=('gzip', 'lzf', 'none'), default='lzf', help='compression algorithm')
parser.add_argument('-s', '--split', action='store_true', default=False, help='split output file')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='debugging')
#parser.add_argument('--precision', action='store', type=int, choices=(8,16,32,64), default=32, help='Precision')
parser.add_argument('--deltaR', action='store', type=float, default=1.2, help='maximum deltaR to build graphs')
args = parser.parse_args()

if not args.output.endswith('.h5'): outPrefix, outSuffix = args.output+'/data', '.h5'
else: outPrefix, outSuffix = args.output.rsplit('.', 1)
args.nevent = max(args.nevent, -1) ## nevent should be -1 to process everything or give specific value
#precision = 'f%d' % (args.precision//8)
#kwargs = {'dtype':precision}
#if args.compress == 'gzip':
#    kwargs.update({'compression':'gzip', 'compression_opts':9})
#elif args.compress == 'lzf':
#    kwargs.update({'compression':'lzf'})

treeName = "Delphes" #if args.format == "Delphes" else "Events"
weightName = None
if not args.data:
    weightName = "Weight"
    #if args.format == "Delphes" else "weights"

## Logic for the arguments regarding on splitting
##   split off:
##     nevent == -1: process all events store in one file
##     nevent != -1: process portion of events, store in one file
##   split on:
##     nevent == -1: process all events, split into nfiles
##     nevent != -1: split files, limit total number of events to be nevent
##     nevent != -1: split files by nevents for each files

## Find root files with corresponding trees
print("@@@ Checking input files... (total %d files)" % (len(args.input)))
nEventTotal = 0
nEvent0s = []
srcFileNames = []
for x in args.input:
    for fName in glob(x):
        if not fName.endswith('.root'): continue
        f = uproot.open(fName)
        if treeName not in f: continue
        tree = f[treeName]
        if tree == None: continue

        if args.debug and nEventTotal == 0:
            print("-"*40)
            print("\t".join([str(key) for key in tree.keys()]))
            print("\t".join([str(key) for key in tree["Jet"].keys()]))
            print("-"*40)

        srcFileNames.append(fName)
        nEvent0 = len(tree)
        nEvent0s.append(nEvent0)
        nEventTotal += nEvent0
nEventOutFile = min(nEventTotal, args.nevent) if args.split else nEventTotal
print("@@@ Total %d events to process, store %d events per file" % (nEventTotal, nEventOutFile))

maxDR2 = args.deltaR*args.deltaR ## maximum deltaR value to connect two jets
@numba.njit(nogil=True, fastmath=True, parallel=True)
def buildGraph(jetss_pt, jetss_eta, jetss_phi):
    prange = numba.prange

    nodes1, nodes2 = [[0]], [[0]]
    nodes1.pop()
    nodes2.pop()
    nEvent = len(jetss_eta)
    for ievt in prange(nEvent):
        selJets = (jetss_pt[ievt] > 30) & (np.fabs(jetss_eta[ievt]) < 2.4)
        jets_eta = jetss_eta[ievt][selJets]
        jets_phi = jetss_phi[ievt][selJets]
        nJet = len(jets_eta)

        inodes1, inodes2 = [], []
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
                inodes1.append(i)
                inodes2.append(j)
        nodes1.append(inodes1+inodes2)
        nodes2.append(inodes2+inodes1)

    return nodes1, nodes2

@numba.njit(nogil=True, fastmath=True, parallel=True)
def selectBaselineCuts(src_fjets_pt, src_fjets_eta, src_fjets_mass,
                       src_jets_pt, src_jets_eta, src_jets_btag):
    nEvent = int(len(src_fjets_pt))
    selEvents = []

    prange = numba.prange
    for ievt in prange(nEvent):
        selJets = (src_jets_pt[ievt] > 30) & (np.fabs(src_jets_eta[ievt]) < 2.4)
        if selJets.sum() < 4: continue ## require nJets >= 4
        ht = (src_jets_pt[ievt][selJets]).sum()
        if ht < 1500: continue ## require HT >= 1500

        selBJets = (src_jets_btag[ievt][selJets] > 0.5)
        if selBJets.sum() < 1: continue ## require nBJets >= 1

        selFjets = (src_fjets_pt[ievt] > 30)
        sumFjetsMass = (src_fjets_mass[ievt][selFjets]).sum()
        if sumFjetsMass < 500: continue ## require sum(FatJetMass) >= 500

        selEvents.append(ievt)

    return np.array(selEvents, dtype=np.dtype('int64'))

class FileSplitOut:
    def __init__(self, maxEvent, featNames, fNamePrefix, chunkSize, debug=False):
        self.maxEvent = maxEvent
        self.featNames = featNames
        self.fNamePrefix = fNamePrefix
        self.chunkSize = chunkSize
        self.debug = debug

        self.type_fa = h5py.special_dtype(vlen=np.dtype('float64'))
        self.type_ia = h5py.special_dtype(vlen=np.dtype('uint32'))
        self.nOutFile = 0
        self.nOutEvent = 0

        self.initOutput()

    def initOutput(self):
        ## Build placeholder for the output
        self.weights = np.ndarray((0,), dtype=np.dtype('float64'))
        self.jets_eta = np.ndarray((0,), dtype=self.type_fa)
        self.jets_phi = np.ndarray((0,), dtype=self.type_fa)
        self.jets_feats = [np.ndarray((0,), dtype=self.type_fa) for _ in self.featNames]
        self.jets_node1 = np.ndarray((0,), dtype=self.type_ia)
        self.jets_node2 = np.ndarray((0,), dtype=self.type_ia)

    def addEvents(self, src_weights, src_jets_eta, src_jets_phi, src_jets_feats,
                        jets_node1, jets_node2):
        nSrcEvent = len(src_weights)
        self.nOutEvent += nSrcEvent;
        begin = 0
        while begin < nSrcEvent:
            end = begin+min(self.maxEvent, nSrcEvent)

            self.weights = np.concatenate([self.weights, src_weights[begin:end]])
            self.jets_eta = self.join(self.jets_eta, src_jets_eta[begin:end])
            self.jets_phi = self.join(self.jets_phi, src_jets_phi[begin:end])
            for i in range(nFeats):
                self.jets_feats[i] = self.join(self.jets_feats[i], src_jets_feats[i][begin:end])
            self.jets_node1 = self.join(self.jets_node1, jets_node1[begin:end])
            self.jets_node2 = self.join(self.jets_node2, jets_node2[begin:end])

            if len(self.weights) == self.maxEvent: self.flush()
            begin = end

    def flush(self):
        self.save()
        self.initOutput()

    def save(self):
        fName = "%s_%d.h5" % (self.fNamePrefix, self.nOutFile)
        nEventToSave = len(self.weights)
        if nEventToSave == 0: return

        with h5py.File(fName, 'w', libver='latest', swmr=True) as outFile:
            out_events = outFile.create_group('events')
            out_events.create_dataset('weights', data=self.weights, chunks=(self.chunkSize,), dtype='f4')

            out_jets = outFile.create_group('jets')
            out_jets.create_dataset('eta', data=self.jets_eta)
            out_jets.create_dataset('phi', data=self.jets_phi)

            for i, featName in enumerate(featNames):
                shortName = featName.replace('Jet.', '')
                out_jets.create_dataset(shortName, (nEventToSave,), dtype=self.type_fa)
                out_jets[shortName][...] = self.jets_feats[i]

            out_graphs = outFile.create_group('graphs')
            out_graphs.create_dataset('nodes1', data=self.jets_node1)
            out_graphs.create_dataset('nodes2', data=self.jets_node2)

        self.nOutFile += 1

        if self.debug:
            with h5py.File(fName, 'r', libver='latest', swmr=True) as outFile:
                print("  created %s %dth file" % (fName, self.nOutFile), end='')
                print("  keys=", list(outFile['jets'].keys()), end='')
                print("  shape=", outFile['jets/eta'].shape)
                if len(outFile['events/weights']) > 0:
                    print("    weight[0]=", outFile['events/weights'][0])
                    print("    eta[0]   =", outFile['jets/eta'][0])
                    print("    phi[0]   =", outFile['jets/phi'][0])
                    print("    nodes1[0]=", outFile['graphs/nodes1'][0])
                    print("    nodes2[0]=", outFile['graphs/nodes2'][0])

    def join(self, target, src):
        if len(src) == 0: return target

        ## Add dummy ndarray front pop it, unless numpy build array with wrong dimension.
        arr = [np.array([0])] + [x for x in src]
        arr = np.array(arr, dtype=target.dtype)[1:]
        return np.concatenate([target, arr])
    
print("@@@ Start processing...")

featNames = ("Jet.PT", "Jet.Mass", "Jet.BTag")
nFeats = len(featNames)
fileOuts = FileSplitOut(nEventOutFile, featNames, outPrefix, args.chunk, args.debug)
for nEvent0, srcFileName in zip(nEvent0s, srcFileNames):
    if args.debug: print("@@@ Open file", srcFileName)
    ## Open data files
    fin = uproot.open(srcFileName)
    tree = fin[treeName]

    ## Load objects
    src_weights = np.ones(nEvent0) if weightName else tree[weightName]
    src_jets_eta = tree["Jet"]["Jet.Eta"].array()
    src_jets_phi = tree["Jet"]["Jet.Phi"].array()
    src_jets_feats = [tree["Jet"][featName].array() for featName in featNames]

    ## Apply event selection in this file
    src_fjets_pt   = tree["FatJet"]["FatJet.PT"].array()
    src_fjets_eta  = tree["FatJet"]["FatJet.Eta"].array()
    src_fjets_mass = tree["FatJet"]["FatJet.Mass"].array()
    src_jets_pt   = tree["Jet"]["Jet.PT"].array()
    src_jets_btag = tree["Jet"]["Jet.BTag"].array()
    selEvent = selectBaselineCuts(src_fjets_pt, src_fjets_eta, src_fjets_mass,
                                  src_jets_pt, src_jets_eta, src_jets_btag)

    src_weights = src_weights[selEvent]
    src_jets_pt = src_jets_pt[selEvent]
    src_jets_eta = src_jets_eta[selEvent]
    src_jets_phi = src_jets_phi[selEvent]
    for i in range(nFeats): src_jets_feats[i] = src_jets_feats[i][selEvent]

    ## Build graphs
    jets_node1, jets_node2 = buildGraph(src_jets_pt, src_jets_eta, src_jets_phi)
    if args.debug:
        if len(jets_node1) > 0:
            print("@@@ Debug: First graphs built:", jets_node1[0], jets_node2[0])
            print("           eta:", src_jets_eta[0])
            print("           phi:", src_jets_phi[0])
        else:
            print("@@@ Debug: Graphs in the first input file is empty...")
    jets_node1 = [np.array(x, dtype=np.uint32) for x in jets_node1]
    jets_node2 = [np.array(x, dtype=np.uint32) for x in jets_node2]

    ## Save output
    fileOuts.addEvents(src_weights, src_jets_eta, src_jets_phi, src_jets_feats, jets_node1, jets_node2)
## save remaining events
fileOuts.flush()

print("@@@ Finished processing")
print("    Number of input files   =", len(srcFileNames))
print("    Number of input events  =", nEventTotal)
print("    Number of output files  =", fileOuts.nOutFile)
print("    Number of output events =", fileOuts.nOutEvent)
