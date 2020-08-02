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
parser.add_argument('--nfiles', action='store', type=int, default=0, help='number of output files')
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
##   split off: we will simply ignore nfiles parameter => reset nfiles=1
##     nevent == -1: process all events store in one file
##     nevent != -1: process portion of events, store in one file
##   split on:
##     nevent == -1, nfiles == 1: same as the one without splitting
##     nevent != -1, nfiles == 1: same as the one without splitting
##     nevent == -1, nfiles != 1: process all events, split into nfiles
##     nevent != -1, nfiles != 1: split files, limit total number of events to be nevent
##     nevent != -1, nfiles == 0: split files by nevents for each files
if not args.split or args.nfiles == 1:
    ## Just confirm the options for no-splitting case
    args.split = False
    args.nfiles = 1
elif args.split and args.nevent > 0:
    args.nfiles = 0

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
if args.nfiles > 0:
    nEventOutFile = int(ceil(nEventTotal/args.nfiles))
else:
    args.nfiles = int(ceil(nEventTotal/args.nevent))
    nEventOutFile = min(nEventTotal, args.nevent)
print("@@@ Total %d events to process, store into %d files (%d events per file)" % (nEventTotal, args.nfiles, nEventOutFile))

@numba.njit(nogil=True, fastmath=True)
def buildGraph(jetss_eta, jetss_phi):
    prange = numba.prange
    maxDR2 = 1.2*1.2 ## maximum deltaR value to connect two jets

    nodes1, nodes2 = [[0]], [[0]]
    nodes1.pop()
    nodes2.pop()
    nEvent = len(jetss_eta)
    for ievt in range(nEvent):
        jets_eta = jetss_eta[ievt]
        jets_phi = jetss_phi[ievt]
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

@numba.jit(nogil=True, fastmath=True)
def selectBaselineCuts(fjetss_pt, fjetss_eta, fjetss_mass,
                       jetss_pt, jetss_eta, jetss_btag):
    prange = numba.prange

    cut_minFJetSumMass = 500 # GeV
    cut_minNBJets = 1
    cut_minHT = 1500 ## GeV
    cut_minNJet = 4

    fatjet_pt_min = 30*units.GeV
    jet_pt_min = 30*units.GeV
    jet_eta_max = 2.4

    return True

print("@@@ Start processing...")

outFileNames = []
nEventProcessed = 0
nEventToGo = nEventOutFile

featNames = ("Jet.PT", "Jet.Mass", "Jet.BTag")
nFeats = len(featNames)
dtype = h5py.special_dtype(vlen=np.dtype('float64'))
itype = h5py.special_dtype(vlen=np.dtype('uint32'))

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

    ## Analyze events in this file
    nEventPassed = nEvent0 ## FIXME: to be changed to cound number of events after cuts

    jets_node1, jets_node2 = buildGraph(src_jets_eta, src_jets_phi)
    if args.debug:
        print("@@@ Debug: First graphs built:", jets_node1[0], jets_node2[0])
        print("           eta:", src_jets_eta[0])
        print("           phi:", src_jets_phi[0])

    begin, end = 0, min(nEventToGo, nEvent0)
    while begin < nEvent0:
        ### First check to prepare output array
        if nEventToGo == nEventOutFile: ## Initializing output file
            ## Build placeholder for the output
            out_weights = np.array([], dtype=np.dtype('float64'))
            out_jets_eta = np.array([], dtype=dtype)
            out_jets_phi = np.array([], dtype=dtype)
            out_jets_feats = [np.array([], dtype=dtype) for featName in featNames]
            out_jets_node1 = np.array([], dtype=itype)
            out_jets_node2 = np.array([], dtype=itype)
        ###

        ## Do the processing
        nEventToGo -= (end-begin)
        nEventProcessed += (end-begin)

        out_weights = np.concatenate([out_weights, src_weights[begin:end]])
        out_jets_eta = np.concatenate([out_jets_eta, np.array([list(x) for x in src_jets_eta[begin:end]], dtype=dtype)])
        out_jets_phi = np.concatenate([out_jets_phi, np.array([list(x) for x in src_jets_phi[begin:end]], dtype=dtype)])
        for i in range(nFeats):
            out_jets_feats[i] = np.concatenate([out_jets_feats[i], 
                                                np.array([list(x) for x in src_jets_feats[i][begin:end]], dtype=dtype)])
        out_jets_node1 = np.concatenate([out_jets_node1, np.array([list(x) for x in jets_node1[begin:end]], dtype=itype)])
        out_jets_node2 = np.concatenate([out_jets_node2, np.array([list(x) for x in jets_node2[begin:end]], dtype=itype)])

        begin, end = end, min(nEventToGo, nEvent0)

        if nEventToGo == 0 or nEventProcessed == nEventTotal: ## Flush output and continue
            nEventToGo = nEventOutFile
            end = min(begin+nEventToGo, nEvent0)

            iOutFile = len(outFileNames)+1
            outFileName = outPrefix + (("_%d" % iOutFile) if args.split else "") + ".h5"
            outFileNames.append(outFileName)
            if args.debug: print("Writing output file %s..." % outFileName, end='')

            chunkSize = min(args.chunk, out_weights.shape[0])
            with h5py.File(outFileName, 'w', libver='latest', swmr=True) as outFile:
                nEventToSave = len(out_weights)
                out = outFile.create_group('jets')

                out.create_dataset('weights', data=out_weights, chunks=(chunkSize,), dtype='f4')

                out.create_dataset('eta', (nEventToSave,), dtype=dtype)
                out.create_dataset('phi', (nEventToSave,), dtype=dtype)
                out['eta'][...] = out_jets_eta
                out['phi'][...] = out_jets_phi

                for i, featName in enumerate(featNames):
                    shortName = featName.replace('Jet.', '')
                    out.create_dataset(shortName, (nEventToSave,), dtype=dtype)
                    out[shortName][...] = out_jets_feats[i]

                out.create_dataset('nodes1', (nEventToSave,), dtype=itype)
                out.create_dataset('nodes2', (nEventToSave,), dtype=itype)
                out['nodes1'][...] = out_jets_node1
                out['nodes2'][...] = out_jets_node2

                if args.debug: print("  done")

            with h5py.File(outFileName, 'r', libver='latest', swmr=True) as outFile:
                print(("  created %s (%d/%d)" % (outFileName, iOutFile, args.nfiles)), end='')
                print("  keys=", list(outFile.keys()), end='')
                print("  shape=", outFile['jets']['eta'].shape)

        print("%d/%d" % (nEventProcessed, nEventTotal), end="\r")

print("done %d/%d" % (nEventProcessed, nEventTotal))

