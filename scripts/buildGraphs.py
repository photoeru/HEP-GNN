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
parser.add_argument('--compress', action='store', choices=('gzip', 'lzf', 'none'), default='lzf', help='compression algorithm')
parser.add_argument('-s', '--split', action='store_true', default=False, help='split output file')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='debugging')
parser.add_argument('--precision', action='store', choices=(8,16,32,64), default=32, help='Precision')
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

    nodes1, nodes2= [[0]], [[0]]
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
out_labels, out_weights, out_image = None, None, None
for iSrcFile, (nEvent0, srcFileName) in enumerate(zip(nEvent0s, srcFileNames)):
    if args.debug: print("@@@ Open file", srcFileName)
    ## Open data files
    fin = uproot.open(srcFileName)
    tree = fin[treeName]
    nEvent = len(tree)

    ## Load objects
    weights = np.ones(nEvent) if weightName else tree[weightName]
    jetss_eta = tree["Jet"]["Jet.Eta"].array()
    jetss_phi = tree["Jet"]["Jet.Phi"].array()
    featNames = ("Jet.PT", "Jet.Mass", "Jet.BTag")
    nFeats = len(featNames)
    jetss_feats = [tree["Jet"][x].array() for x in featNames]

    nEventPassed = nEvent ## FIXME: to be changed to cound number of events after cuts

    nodes1, nodes2 = buildGraph(jetss_eta, jetss_phi)
    if args.debug:
        print("@@@ Debug: First graphs built:", nodes1[0], nodes2[0])
        print("           eta:", jetss_eta[0])
        print("           phi:", jetss_phi[0])

    outFileName = args.output+'/'+srcFileName.rsplit('/',1)[-1].rsplit('.',1)[-1]+'.h5'
    fout = h5py.File(outFileName, mode='w', libver='latest')

    dtype = h5py.special_dtype(vlen=np.dtype('float64'))
    itype = h5py.special_dtype(vlen=np.dtype('uint32'))

    out = fout.create_group('jets')

    out.create_dataset('eta', (nEventPassed,), dtype=dtype)
    out.create_dataset('phi', (nEventPassed,), dtype=dtype)
    out['eta'][...] = np.array([list(x) for x in jetss_eta], dtype=dtype)
    out['phi'][...] = np.array([list(x) for x in jetss_phi], dtype=dtype)
    for i, x in enumerate(featNames):
        shortName = x.replace('Jet.', '')
        out.create_dataset(shortName, (nEventPassed,), dtype=dtype)
        out[shortName][...] = np.array([list(y) for y in jetss_feats[i]], dtype=dtype)

    out.create_dataset('nodes1', (nEventPassed,), dtype=itype)
    out.create_dataset('nodes2', (nEventPassed,), dtype=itype)
    out['nodes1'][...] = np.array([list(x) for x in nodes1], dtype=itype)
    out['nodes2'][...] = np.array([list(x) for x in nodes2], dtype=itype)

    fout.close()

    nEventProcessed += nEvent
    print("%d/%d" % (nEventProcessed, nEventTotal), end="\r")
print("done %d/%d" % (nEventProcessed, nEventTotal))

