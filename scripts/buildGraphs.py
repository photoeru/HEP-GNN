#!/usr/bin/env python3
import argparse
#import h5py
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
parser.add_argument('-o', '--output', action='store', type=str, help='output file name', required=True)
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

## Define variables to save
jetVarNames = ["Jet.BTag"]
nVars = len(jetVarNames)

@numba.njit(nogil=True)#,parallel=True)
def buildGraph(jetss_eta, jetss_phi):
    maxDR2 = 1.2*1.2 ## maximum deltaR value to connect two jets

    graphs = []
    nEvent = len(jetss_eta)
    for ievt in range(nEvent):
        jets_eta = jetss_eta[ievt]
        jets_phi = jetss_phi[ievt]
        nJet = len(jets_eta)

        g = []
        for i in numba.prange(nJet):
            for j in numba.prange(i):
                dEta = jets_eta[i]-jets_eta[j]
                dPhi = jets_phi[i]-jets_phi[j]
                ## Move dPhi to [-pi,pi] range
                if   dPhi >= math.pi: dPhi -= 2*math.pi
                elif dPhi < -math.pi: dPhi += 2*math.pi
                ## Compute deltaR^2 and ask it is inside of our ball
                dR2 = dEta*dEta + dPhi*dPhi
                if dR2 > maxDR2: continue 
                g.append([i,j])
                #g.append([j,i])
        graphs.append(g)

    return graphs

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
    jetss_p4 = [tree["Jet"][x].array() for x in ("Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass")]
    jetss_feats = [tree["Jet"][x].array() for x in jetVarNames]

    g = buildGraph(jetss_p4[1], jetss_p4[2])
    if args.debug:
        print("@@@ Debug: First graphs built:", g[0])
        print("           eta:", jetss_p4[1][0])
        print("           phi:", jetss_p4[2][0])

    nEventProcessed += len(g)
    print("%d/%d" % (nEventProcessed, nEventTotal), end="\r")
print("done %d/%d" % (nEventProcessed, nEventTotal))

