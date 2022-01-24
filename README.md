# L1EGRates

## Setup

Initial setup:

```
cmsrel CMSSW_12_0_0
cd CMSSW_12_0_0/src

git clone https://github.com/MatthewDKnight/L1EGRates.git
cd L1EGRates

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
deactivate

cmsenv

source env/bin/activate
pip install xrootd==5.4.0
```

When using the code, make sure that you `cmsenv` and then `source env/bin/activate` first. The order is important!

```
cmsenv
source env/bin/activate
```

## Usage

Calculates L1 trigger rates.

Inputs:
- List of [L1Ntuples](https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToL1TriggerMenu#4_1_L1Ntuples_for_menu_studies) (data or MC)
- Prescale (PS) [table](https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToL1TriggerMenu#4_3_Make_ntuple_list_Lumi_Sectio)*
- Lumi section [table](https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToL1TriggerMenu#4_3_Make_ntuple_list_Lumi_Sectio)*

*only needed for data

A prescale table and a lumi section table are provided in the repository but you may need to create your own. Check out the Menu Team [twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToL1TriggerMenu#4_3_Make_ntuple_list_Lumi_Sectio).

Outputs:
- Rate for each L1 EG seed
- Plot of rate vs ET

Lists of the most recent (as of 24/01/2022) ntuples produced by the Menu Team are available in `L1EGRates/ntupleLists/`. Check the Menu Team [twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToL1TriggerMenu#4_1_L1Ntuples_for_menu_studies) for my up-to-date versions.

The code is contained in `rate_calculation.py`.

Create some useful directories:
```
mkdir ntuplePkls rateJsons
```

### Running over data

```
python rate_calculation.py --NTuplesList ntupleLists/Run2_Data_RunA_NTuples_MenuTeam.list \
--SaveToDataFrame ntuplePkls/Run2_Data_RunA_NTuples_MenuTeam.pkl \
--PSTable Prescale_2022_v0_1_1.csv --SelectPSFactor 1 \
--OutputJson rateJsons/Run2_Data_RunA_NTuples_MenuTeam.json \
--Data --LumiTable run_lumi_RunA_RunD.csv
```

### Running over MC

```
python rate_calculation.py --NTuplesList ntupleLists/Run3_NuGun_MC_NTuples_MenuTeam.list \
--SaveToDataFrame ntuplePkls/Run3_NuGun_MC_NTuples_MenuTeam.pkl \
--PSTable Prescale_2022_v0_1_1.csv --SelectPSFactor 1 \
--OutputJson rateJsons/Run3_NuGun_MC_NTuples_MenuTeam.json \
--MC
```

The code saves a Pandas DataFrame in the form of a pickle file. This file contains all the information neccessary to calculate the rates. Reloading this DataFrame is far quicker than re-reading the ntuples. You may for example, want to calculate rates for different pileup (PU) windows:

```
python rate_calculation.py --LoadDataFrame ntuplePkls/Run3_NuGun_MC_NTuples_MenuTeam.pkl \
--PSTable Prescale_2022_v0_1_1.csv --SelectPSFactor 1 \
--OutputJson rateJsons/Run3_NuGun_MC_NTuples_MenuTeam_PU_50_56.json \
--MC --PURange 50,56 
```

The `--LowestEtThreshold` option **may** improve processing time and **will** reduce the amount of memory needed. If you are only worried about calculating seed rates, the suggestion would be to set the threshold at 8 GeV (lowest seed threshold). If you want to plot the rate vs ET graph, then you could use the `--NEvents` tag to get a quicker result if need be.

The full list of options:
- `--NTuplesList` Path to a file containing list of NTuple root files.
-  `--NFiles` Specify the number of files in the NTuples list to read. By default all files are read.
-  `--NEvents` Specify the number of events to read in. By default all events are read.
-  `--SaveToDataFrame` Path to a pkl file to save the NTuples as a DataFrame.
-  `--LoadDataFrame` Path to a pkl file containing NTuples as a DataFrame. Script will load this instead of reading root files.
-  `--PSTable` Path to a prescale table in csv format. See [here](https://github.com/cms-l1-dpg/L1MenuTools/blob/master/rate-estimation/menu/Prescale_2022_v0_1_1.csv) for example.
- `--SelectCol` Select a luminosity column in the prescale table.
- `--SelectPSFactor` Select seeds with only the provided prescale factor.
-  `--OutputJson`
-  `--PURange` Pick a PU range, e.g. --PURange 48,56.
-  `--MC` Use if running on MC.
-  `--Data` Use if running on data.
-  `--LumiTable` Path to the lumi section table. Needed when selecting a PU range when running over data.
-  `--LowestEtThreshold` No objects below this ET threshold will be read to save memory and time. Default is 0
- `--NoTrimObjects` By default, the script will only read in the 3 leading objects. If needed, use this option to remove this behaviour.

