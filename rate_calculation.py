import uproot
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict as od
import json
import warnings
import awkward as ak
import pickle
from tqdm import tqdm

import time

seed_dict = od()

#seed_dict['seed_name'] = [nEGs, [ET thresholds], [iso requirements], [eta limits]]
seed_dict['L1_SingleEG8er2p5'] = [1, [16], [0], [-58, 57]]
seed_dict['L1_SingleEG10er2p5'] = [1, [20], [0], [-58, 57]]
seed_dict['L1_SingleEG15er2p5'] = [1, [30], [0], [-58, 57]]
seed_dict['L1_SingleEG26er2p5'] = [1, [32], [0], [-58, 57]]
seed_dict['L1_SingleEG28er2p5'] = [1, [56], [0], [-58, 57]]
seed_dict['L1_SingleEG28er2p1'] = [1, [56], [0], [-49, 48]]
seed_dict['L1_SingleEG28er1p5'] = [1, [56], [0], [-35, 34]]
seed_dict['L1_SingleEG34er2p5'] = [1, [68], [0], [-58, 57]]
seed_dict['L1_SingleEG36er2p5'] = [1, [72], [0], [-58, 57]]
seed_dict['L1_SingleEG38er2p5'] = [1, [76], [0], [-58, 57]]
seed_dict['L1_SingleEG40er2p5'] = [1, [80], [0], [-58, 57]]
seed_dict['L1_SingleEG42er2p5'] = [1, [84], [0], [-58, 57]]
seed_dict['L1_SingleEG45er2p5'] = [1, [90], [0], [-58, 57]]
seed_dict['L1_SingleEG50'] = [1, [100], [0], None]
seed_dict['L1_SingleEG60'] = [1, [120], [0], None]
seed_dict['L1_SingleLooseIsoEG26er2p5'] = [1, [52], [2], [-58, 57]]
seed_dict['L1_SingleLooseIsoEG26er1p5'] = [1, [52], [2], [-35, 34]]
seed_dict['L1_SingleLooseIsoEG28er2p5'] = [1, [56], [2], [-58, 57]]
seed_dict['L1_SingleLooseIsoEG28er2p1'] = [1, [56], [2], [-49, 48]]
seed_dict['L1_SingleLooseIsoEG28er1p5'] = [1, [56], [2], [-35, 34]]
seed_dict['L1_SingleLooseIsoEG30er2p5'] = [1, [60], [2], [-58, 57]]
seed_dict['L1_SingleLooseIsoEG30er1p5'] = [1, [60], [2], [-35, 34]]
seed_dict['L1_SingleIsoEG24er2p1'] = [1, [48], [1], [-49, 48]]
seed_dict['L1_SingleIsoEG24er1p5'] = [1, [48], [1], [-35, 34]]
seed_dict['L1_SingleIsoEG26er2p5'] = [1, [52], [1], [-58, 57]]
seed_dict['L1_SingleIsoEG26er2p1'] = [1, [52], [1], [-49, 48]]
seed_dict['L1_SingleIsoEG26er1p5'] = [1, [52], [1], [-35, 34]]
seed_dict['L1_SingleIsoEG28er2p5'] = [1, [56], [1], [-58, 57]]
seed_dict['L1_SingleIsoEG28er2p1'] = [1, [56], [1], [-49, 48]]
seed_dict['L1_SingleIsoEG28er1p5'] = [1, [56], [1], [-35, 34]]
seed_dict['L1_SingleIsoEG30er2p5'] = [1, [60], [1], [-58, 57]]
seed_dict['L1_SingleIsoEG30er2p1'] = [1, [60], [1], [-49, 48]]
seed_dict['L1_SingleIsoEG32er2p5'] = [1, [64], [1], [-58, 57]]
seed_dict['L1_SingleIsoEG32er2p1'] = [1, [64], [1], [-49, 48]]
seed_dict['L1_SingleIsoEG34er2p5'] = [1, [68], [1], [-58, 57]]
seed_dict['L1_DoubleEG_15_10_er2p5'] = [2, [30,20], [0,0], [-58, 57]]
seed_dict['L1_DoubleEG_20_10_er2p5'] = [2, [40,20], [0,0], [-58, 57]]
seed_dict['L1_DoubleEG_22_10_er2p5'] = [2, [44,20], [0,0], [-58, 57]]
seed_dict['L1_DoubleEG_25_12_er2p5'] = [2, [50,24], [0,0], [-58, 57]]
seed_dict['L1_DoubleEG_25_14_er2p5'] = [2, [50,28], [0,0], [-58, 57]]
seed_dict['L1_DoubleEG_27_14_er2p5'] = [2, [54,28], [0,0], [-58, 57]]
seed_dict['L1_DoubleEG_LooseIso20_10_er2p5'] = [2, [40,20], [2,0], [-58, 57]]
seed_dict['L1_DoubleEG_LooseIso22_10_er2p5'] = [2, [44,20], [2,0], [-58, 57]]
seed_dict['L1_DoubleEG_LooseIso22_12_er2p5'] = [2, [44,24], [2,0], [-58, 57]]
seed_dict['L1_DoubleEG_LooseIso25_12_er2p5'] = [2, [50,24], [2,0], [-58, 57]]
seed_dict['L1_DoubleLooseIsoEG22er2p1'] = [2, [44,44], [2,2], [-49, 48]]
seed_dict['L1_DoubleLooseIsoEG24er2p1'] = [2, [48,48], [2,2], [-49, 48]]
seed_dict['L1_TripleEG_16_12_8_er2p5'] = [3, [32,24,16], [0,0,0], [-58, 57]]
seed_dict['L1_TripleEG_16_15_8_er2p5'] = [3, [32,30,16], [0,0,0], [-58, 57]]
seed_dict['L1_TripleEG_18_17_8_er2p5'] = [3, [36,34,16], [0,0,0], [-58, 57]]
seed_dict['L1_TripleEG_18_18_12_er2p5'] = [3, [36,36,24], [0,0,0], [-58, 57]]
seed_dict['L1_TripleEG16er2p5'] = [3, [32,32,32], [0,0,0], [-58, 57]]

seeds = seed_dict.keys()

def calculateFraction(eg, nEvents, nEG, Et_thresholds, Iso_thresholds, eta_threshold=None):  
  #prepare cut on eta
  if eta_threshold==None:
    eta_cut = np.ones(len(eg), dtype='bool')
  else:
    eta_cut = (eta_threshold[0] <= eg.egIEta) & (eg.egIEta <= eta_threshold[1])
  
  #preliminary cut based on loosest Eta and Iso requirements
  et_cut = eg.egIEt >= Et_thresholds[-1]
  #iso_cut = eg.egIso >= Iso_thresholds[-1]
  iso_cut = eg.egIso >= 0

  combined_cut = eta_cut & et_cut & iso_cut
  eta = eg.egIEta[combined_cut]
  et = eg.egIEt[combined_cut]
  iso = eg.egIso[combined_cut]

  conditions = []
  for i in range(nEG-1, -1, -1):
    et_cut = et >= Et_thresholds[i]
    #true if (no iso requirement) or (meets both loose and tight) or (meets specified requirement)
    iso_cut = (Iso_thresholds[i]==0) | (iso==3) | (iso == Iso_thresholds[i])
    conditions.append( (et_cut & iso_cut).sum(level=0) > i ) #greater than i particles satisfying these conditions
      
  condition = conditions[0]
  for c in conditions[1:]:
    condition = condition & c

  count = np.sum(condition)
  fraction = float(count) / nEvents
  fraction_err = np.sqrt(count) / nEvents

  return fraction, fraction_err

def etPlot(eg, nEvents):
  nBins = 50
  thresholds = np.arange(nBins)
  Ithresholds = thresholds*2 #because IEt = 2*Et
  scale = (11246*2544) / 1000

  rates = []
  for t in tqdm(Ithresholds):
    fraction, err = calculateFraction(eg, nEvents, 1, [t], [0], None)
    rates.append(fraction*scale)
  plt.hist(thresholds, nBins, weights=rates, histtype='step', label="SingleEG Inclusive")

  rates = []
  for t in tqdm(Ithresholds):
    fraction, err = calculateFraction(eg, nEvents, 1, [t], [1], None)
    rates.append(fraction*scale)
  plt.hist(thresholds, nBins, weights=rates, histtype='step', label="SingleEG TightIso")

  rates = []
  for t in tqdm(Ithresholds):
    fraction, err = calculateFraction(eg, nEvents, 2, [t+10*2, t], [0, 0], None)
    rates.append(fraction*scale)
  plt.hist(thresholds, nBins, weights=rates, histtype='step', label="DoubleEG Inclusive")

  rates = []
  for t in tqdm(Ithresholds):
    fraction, err = calculateFraction(eg, nEvents, 2, [t+10*2, t], [2, 0], None)
    rates.append(fraction*scale)
  plt.hist(thresholds, nBins, weights=rates, histtype='step', label="DoubleEG LooseIso")

  plt.yscale("log")
  plt.ylabel("Rate (kHz)")
  plt.xlabel(r"$E_T$")
  plt.legend()
  plt.savefig("rate_py.png")
  plt.savefig("rate_py.pdf")

def calculateSeedRates(eg, nEvents, ps_dict, options):
  print(">> Calculating seed rates")
  rate_dict = od()
  
  for seed in seeds:
    if options.ps_factor != None:
      if ps_dict[seed] != options.ps_factor: continue

    nEG, Et_thresholds, Iso_thresholds, eta_threshold = seed_dict[seed]
    fraction, err = calculateFraction(eg, nEvents, nEG, Et_thresholds, Iso_thresholds, eta_threshold)
    print("> %s: %f +- %f"%(seed, fraction, err))
    scale = (float(11246*2544) / ps_dict[seed]) / 1000
    rate_dict[seed] = fraction * scale
    rate_dict[seed+"_uncert"] = err * scale

  return rate_dict

def getPU(eg, event, options, lumi_table):
  if options.mc:
    return eg, ak.values_astype(event.nPV_True, "uint8")
  else:
    PU = np.zeros(len(event), dtype="float32")
    for run in np.unique(event.run):
      if run not in lumi_table.index:
        PU[event.run==run] = -1
        continue

      for ls in np.unique(event[event.run==run].lumi):
        if (run, ls) in lumi_table.index:
          PU[(event.run==run)&(event.lumi==ls)] = lumi_table.loc[(run,ls)]
        else:
          PU[(event.run==run)&(event.lumi==ls)] = -1

    PU = ak.Array(PU)

    return eg, PU

def readRootFiles(options, lumi_table=None):
  with open(options.ntuple_list, "r") as ntuple_list:
    files = ntuple_list.read().split("\n")
  files = list(filter(lambda x: x != "", files)) #remove empty lines
  if options.n_files != -1: 
    files = files[:options.n_files]
  
  print(">> Reading in %d files"%len(files))
  print("> Head of files list:")
  print(" "+"\n ".join(files[:5]))

  egBranches = ["egIEt", "egIEta", "egIso"]
  if options.mc: eventBranches = ["nPV_True"]
  else:          eventBranches = ["run", "lumi"]

  egArrays = []
  PUArrays = []
  n_events_read = 0
  n_events_kept = 0
  with tqdm(files) as t:
    for filename in t:
      try:
        f = uproot.open(filename, begin_chunk_size=256)
        eg = f["l1UpgradeEmuTree/L1UpgradeTree"].arrays(egBranches, "(egIEt>=%d)&(egBx==0)"%(options.lowest_et_threshold*2))
        if options.trim_objects: eg = eg[:,:3] #keep leading 3 objects
        event = f["l1EventTree/L1EventTree"].arrays(eventBranches)
        if options.data: 
          ZeroBias_idx = int(f["l1uGTTree/L1uGTTree"].aliases["L1_ZeroBias"][-4:-1])
          uGT = f["l1uGTTree/L1uGTTree"].arrays(["decision"], aliases={"decision":"m_algoDecisionFinal[:,%d]"%ZeroBias_idx})
        f.close()

        eg["egIso"] = ak.values_astype(eg["egIso"], "int8") #save some memory

        n_events_read += len(eg)

        #apply cut according to uGT final decision, I'm unsure what the purpose of this is but it's in the Menu Team's code
        if options.data:
          cut = uGT.decision
          eg, event = eg[cut], event[cut]

        n_events_kept += len(eg)
        t.set_postfix(n_read=n_events_read, n_kept=n_events_kept)

        eg, PU = getPU(eg, event, options, lumi_table)
        eg["PU"] = PU
        
        egArrays.append(eg)
        PUArrays.append(PU)
      except Exception as e:
       print(e)
       print("Failed to read %s"%filename)
      if (options.n_events != -1) & (n_events_kept >= options.n_events):
        break

  print("> Kept %d/%d events"%(n_events_kept, n_events_read))

  print("> Concatenating files (DataFrames)")
  eg = ak.to_pandas(ak.concatenate(egArrays))
  PU = pd.Series(ak.to_numpy(ak.concatenate(PUArrays)))
  assert n_events_kept == len(PU)
  print("> Read in %.2fk events"%(float(n_events_read)/1000))
   
  return eg, PU

def getLumiTable(options):
  if options.lumi_table == None:
    return None
  else:
    print(">> Reading %s"%options.lumi_table)
    lumi_table = pd.read_csv(options.lumi_table)
    lumi_table.set_index(["run", "ls"], inplace=True)
    lumi_table = lumi_table[(lumi_table.index.duplicated()) | (~lumi_table.index.duplicated(keep=False))]
    lumi_table.sort_index(inplace=True)
    print(lumi_table)
  return lumi_table["avgpu"]

def getOptions():
  from optparse import OptionParser
  parser = OptionParser(usage="%prog [options]")
  parser.add_option("--NTuplesList", dest="ntuple_list", default=None,
                    help="Path to a file containing list of NTuple root files.")
  parser.add_option("--NFiles", dest="n_files", default=-1, type=int,
                    help="Specify the number of files in the NTuples list to read. By default all files are read.")
  parser.add_option("--NEvents", dest="n_events", default=-1, type=int,
                    help="Specify the number of events to read in. By default all files are read.")
  parser.add_option("--SaveToDataFrame", dest="save_dataframe", default=None,
                    help="Path to a pkl file to save the NTuples as a DataFrame.")
  parser.add_option("--LoadDataFrame", dest="load_dataframe", default=None,
                    help="Path to a pkl file containing NTuples as a DataFrame. Script will load this instead of reading root files.")
  parser.add_option("--PSTable", dest="ps_table", default=None,
                    help="Path to a prescale table in csv format. See https://github.com/cms-l1-dpg/L1MenuTools/blob/master/rate-estimation/menu/Prescale_2022_v0_1_1.csv for example.")
  parser.add_option("--SelectCol", dest="ps_col", default="2E+34",
                    help="Select a luminosity column in the prescale table.")
  parser.add_option("--SelectPSFactor", dest="ps_factor", default=None, type=int,
                    help="Select seeds with only the provided prescale factor.")
  parser.add_option("--OutputJson", dest="output_json", default="seed_rates.json")
  parser.add_option("--PURange", dest="pu_range", default=None,
                    help="Pick a PU range, e.g. --PURange 48,56.")
  parser.add_option("--MC", dest="mc", default=False, action="store_true",
                    help="Use if running on MC.")
  parser.add_option("--Data", dest="data", default=False, action="store_true",
                    help="Use if running on data.")
  parser.add_option("--LumiTable", dest="lumi_table", default=None,
                    help="Path to the lumi section table. Needed when selecting a PU range when running over data.")

  parser.add_option("--LowestEtThreshold", dest="lowest_et_threshold", default=0, type=int,
                    help="No objects below this ET threshold are read to save memory and time. Default is minimum ET from EG seeds (8 GeV)")
  parser.add_option("--NoTrimObjects", dest="trim_objects", default=True, action="store_false",
                    help="By default, the script will only read in the 3 leading objects. If needed, use this option to remove this behaviour.")
  

  (options, args) = parser.parse_args()

  if options.mc==options.data:
    parser.print_help()
    raise Exception("Must provide specify --MC or --Data.")

  if (options.data and (options.pu_range != None)) and (options.lumi_table == None):
    raise Exception("If selecting a PU range for data, a lumi section table is required.")
  elif options.data and (options.lumi_table == None):
    warnings.warn("Without providing a lumi section table, PU information will not be saved in the DataFrame and applying a PU window will require re-reading the root files.")

  if options.mc and (options.lumi_table != None):
    warnings.warn("There is no need for a lumi section table when running on MC. It will not be used.")

  if (options.ntuple_list==None) and (options.load_dataframe==None):
    parser.print_help()
    raise Exception("Must provide a list of NTuples or an equivalent DataFrame (as a pkl file).")
  elif (options.ntuple_list!=None) and (options.load_dataframe!=None):
    parser.print_help()
    raise Exception("Please provide either a list of NTuples or an equivalent DataFrame (as a pkl file). Do not provide both.")

  if (options.ps_table==None):
    warnings.warn("Without a prescale table, a prescale factor of 1 will be assumed for all seeds")

  if (options.ps_factor!=None) and (options.ps_table==None):
    parser.print_help()
    raise Exception("To use the --SelectPSFactor option, a prescale table has to be provided with --PSTable.")
    
  return options

def main(options):
  lumi_table = getLumiTable(options)

  if options.ntuple_list != None:
    eg, PU = readRootFiles(options, lumi_table)
  else:
    print(">> Loading %s"%options.load_dataframe)
    with open(options.load_dataframe, "rb") as f:
      eg, PU = pickle.load(f)

  if options.save_dataframe != None:
    print(">> Saving %s"%options.save_dataframe)
    with open(options.save_dataframe, "wb") as f:
      pickle.dump([eg, PU], f)

  if options.ps_table != None:
    ps_table = pd.read_csv(options.ps_table)
    ps_table = ps_table[["Name", options.ps_col]]
    ps_table.columns = ["Name", "PS"]
  else:
    ps_table = pd.DataFrame({"Name":seeds, "PS": [1 for each in seeds]})
  ps_table.set_index("Name", inplace=True)
  ps_dict = ps_table["PS"].to_dict()

  if options.pu_range != None:
    PU_low, PU_high = options.pu_range.split(",")
    PU_low, PU_high = int(PU_low), int(PU_high)
  else:
    PU_low, PU_high = -1, 9999

  print(">> Performing PU selection: %d <= PU <= %d"%(PU_low, PU_high))
  eg = eg[(eg.PU>=PU_low)&(eg.PU<=PU_high)]
  nEvents = int(((PU>=PU_low)&(PU<=PU_high)).sum())
  print("> After PU selection there are %.2fk events"%(float(nEvents)/1000))  

  rate_dict = calculateSeedRates(eg, nEvents, ps_dict, options)
  with open(options.output_json, "w") as f:
    json.dump(rate_dict, f, indent=4)

  etPlot(eg, nEvents)

  return eg, PU

if __name__=="__main__":
  options = getOptions()
  eg, PU = main(options)