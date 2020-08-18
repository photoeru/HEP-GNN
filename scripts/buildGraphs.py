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
parser.add_argument('--compress', action='store', choices=('gzip', 'lzf', 'none'), default='lzf', help='compression algorithm')
parser.add_argument('-s', '--split', action='store_true', default=False, help='split output file')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='debugging')
parser.add_argument('--precision', action='store', type=int, choices=(8,16,32,64), default=32, help='Precision')
args = parser.parse_args()

if not args.output.endswith('.h5'): outPrefix, outSuffix = args.output+'/data', '.h5'
else: outPrefix, outSuffix = args.output.rsplit('.', 1)
args.nevent = max(args.nevent, -1) ## nevent should be -1 to process everything or give specific value
precision = 'f%d' % (args.precision//8)
kwargs = {'dtype':precision}
if args.compress == 'gzip':
    kwargs.update({'compression':'gzip', 'compression_opts':9})
elif args.compress == 'lzf':
    kwargs.update({'compression':'lzf'})

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

@numba.njit(nogil=True, fastmath=True, parallel=True)
def buildGraph(jetss_pt, jetss_eta, jetss_phi):
    prange = numba.prange
    maxDR2 = 1.2*1.2 ## maximum deltaR value to connect two jets

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
        if len(selJets) < 4: continue ## require nJets >= 4
        ht = (src_jets_pt[ievt][selJets]).sum()
        if ht < 1500: continue ## require HT >= 1500

        selBJets = (src_jets_btag[ievt][selJets] > 0.5)
        if sum(selBJets) < 1: continue ## require nBJets >= 1

        selFjets = (src_fjets_pt[ievt] > 30)
        sumFjetsMass = (src_fjets_mass[ievt][selFjets]).sum()
        if sumFjetsMass < 500: continue ## require sum(FatJetMass) >= 500

        selEvents.append(ievt)

    return np.array(selEvents, dtype=np.dtype('int64'))

print("@@@ Start processing...")

outFileNames = []
nEventProcessed = 0
nEventToGo = nEventOutFile

featNames = ("Jet.PT", "Jet.Mass", "Jet.BTag")
nFeats = len(featNames)
dtype = h5py.special_dtype(vlen=np.dtype('float64'))
itype = h5py.special_dtype(vlen=np.dtype('uint32'))

nSrcFiles = len(srcFileNames)
for iSrcFile, (nEvent0, srcFileName) in enumerate(zip(nEvent0s, srcFileNames)):
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
    src_jets_btag = tree["Jet"]["Jet.Mass"].array()
    selEvent = selectBaselineCuts(src_fjets_pt, src_fjets_eta, src_fjets_mass,
                                  src_jets_pt, src_jets_eta, src_jets_btag)

    nEventPassed = len(selEvent)
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

    begin, end = 0, min(nEventToGo, nEventPassed)
    while begin < nEventPassed: ## Inner loop to store events to splited output files
        ### First check to prepare output array
        if nEventToGo == nEventOutFile: ## Initializing output file
            ## Build placeholder for the output
            out_weights = np.array([], dtype=np.dtype('float64'))
            out_jets_eta = np.array([], dtype=dtype)
            out_jets_phi = np.array([], dtype=dtype)
            out_jets_feats = [np.array([], dtype=dtype) for i in range(nFeats)]
            out_jets_node1 = np.array([], dtype=itype)
            out_jets_node2 = np.array([], dtype=itype)
        ###

        ## Do the processing
        nEventToGo -= (end-begin)

        out_weights = np.concatenate([out_weights, src_weights[begin:end]])
        if (end-begin) <= 1: ## Exceptional case: np.array mistakes output shape to be (1,N) of float type but we need (1,) of list type
            out_jets_eta = np.concatenate([out_jets_eta, np.array([[src_jets_eta[begin]],[]], dtype=dtype)])[:-1]
            out_jets_phi = np.concatenate([out_jets_phi, np.array([[src_jets_phi[begin]],[]], dtype=dtype)])[:-1]
            for i in range(nFeats):
                out_jets_feats[i] = np.concatenate([out_jets_feats[i],
                                                    np.array([[src_jets_feats[i][begin]],[]], dtype=dtype)])[:-1]
            out_jets_node1 = np.concatenate([out_jets_node1, np.array([[jets_node1[begin]],[]], dtype=itype)])[:-1]
            out_jets_node2 = np.concatenate([out_jets_node2, np.array([[jets_node2[begin]],[]], dtype=itype)])[:-1]

        else:
            out_jets_eta = np.concatenate([out_jets_eta, np.array([list(x) for x in src_jets_eta[begin:end]], dtype=dtype)])
            out_jets_phi = np.concatenate([out_jets_phi, np.array([list(x) for x in src_jets_phi[begin:end]], dtype=dtype)])
            for i in range(nFeats):
                out_jets_feats[i] = np.concatenate([out_jets_feats[i],
                                                    np.array([list(x) for x in src_jets_feats[i][begin:end]], dtype=dtype)])
            out_jets_node1 = np.concatenate([out_jets_node1, np.array([list(x) for x in jets_node1[begin:end]], dtype=itype)])
            out_jets_node2 = np.concatenate([out_jets_node2, np.array([list(x) for x in jets_node2[begin:end]], dtype=itype)])

        begin, end = end, min(nEventToGo, nEventPassed)

        if nEventToGo == 0 or (iSrcFile == nSrcFiles-1 and nEventToGo <= nEventOutFile): ## Flush output and continue
            nEventToGo = nEventOutFile
            end = min(begin+nEventToGo, nEventPassed)

            iOutFile = len(outFileNames)+1
            outFileName = outPrefix + (("_%d" % iOutFile) if args.split else "") + ".h5"
            outFileNames.append(outFileName)
            if args.debug: print("Writing output file %s..." % outFileName, end='')

            chunkSize = min(args.chunk, out_weights.shape[0])
            with h5py.File(outFileName, 'w', libver='latest', swmr=True) as outFile:
                nEventToSave = len(out_weights)
                out_events = outFile.create_group('events')
                out_events.create_dataset('weights', data=out_weights, chunks=(chunkSize,), dtype='f4')

                out_jets = outFile.create_group('jets')
                out_jets.create_dataset('eta', (nEventToSave,), dtype=dtype)
                out_jets.create_dataset('phi', (nEventToSave,), dtype=dtype)
                out_jets['eta'][...] = out_jets_eta
                out_jets['phi'][...] = out_jets_phi

                for i, featName in enumerate(featNames):
                    shortName = featName.replace('Jet.', '')
                    out_jets.create_dataset(shortName, (nEventToSave,), dtype=dtype)
                    out_jets[shortName][...] = out_jets_feats[i]

                out_graphs = outFile.create_group('graphs')
                out_graphs.create_dataset('nodes1', (nEventToSave,), dtype=itype)
                out_graphs.create_dataset('nodes2', (nEventToSave,), dtype=itype)
                out_graphs['nodes1'][...] = out_jets_node1
                out_graphs['nodes2'][...] = out_jets_node2

                if args.debug: print("  done")

            with h5py.File(outFileName, 'r', libver='latest', swmr=True) as outFile:
                print("  created %s %dth file" % (outFileName, iOutFile), end='')
                print("  keys=", list(outFile['jets'].keys()), end='')
                print("  shape=", outFile['jets/eta'].shape)

    nEventProcessed += nEvent0
    print("%d/%d" % (nEventProcessed, nEventTotal), end="\r")

print("done %d/%d" % (nEventProcessed, nEventTotal))

