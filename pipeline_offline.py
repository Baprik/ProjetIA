import time
from datetime import datetime
from scapy.all import sniff, wrpcap
from converter import capToCicids2018Linux, capToHikari2021
from loader import loadCapHiraki, loadCapCicids
import os 
from model import MalwareDetector4 
import torch
import pandas as pd 
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument('--cap', type=str, default=None, help='File of the Cap ')
parser.add_argument('--csv', type=str, default=None, help='Optionnal: file of the CSV CCIDS2018')


args = parser.parse_args()

capture_file = args.cap



#### HIRAKI 
capToHikari2021(capture_file)
path =os.path.join("output_Hiraki", f"{capture_file.replace('.pcap', '.csv')}")
X, ids = loadCapHiraki(path)
model_Hiraki = MalwareDetector4(X)
state_dict = torch.load("model/Hiraki/MalwareDectetor4Hiraki_33%.pt")
model_Hiraki.load_state_dict(state_dict)
model_Hiraki.eval()
y_hiraki = model_Hiraki(X)
n = len(y_hiraki)

model_Hiraki_crypto = MalwareDetector4(X)
state_dict = torch.load("model/Hiraki/MalwareDectetor4Hiraki_crypto.pt")
model_Hiraki_crypto.load_state_dict(state_dict)
model_Hiraki_crypto.eval()
y_hiraki_crypto = model_Hiraki_crypto(X)

path =os.path.join("output_Hiraki", f"{capture_file.replace('.pcap', '_conn.csv')}")
log = pd.read_csv(path,skiprows=1)


#### CICIDS: 
"""
capToCicids2018Linux(file + capture_file)
path =os.path.join("output_Cicids", f"{capture_file.replace('.pcap', '.csv')}")
"""
if args.csv != None: 
    X, ids = loadCapCicids(args.csv)
    model_Cicids = MalwareDetector4(X)
    state_dict = torch.load("model/Cicids/MalwareDectetor4Ccids.pt")
    model_Cicids.load_state_dict(state_dict)
    model_Cicids.eval()
    y_cicids = model_Cicids(X)
    n = len(y_cicids)




treshold = 0.4
for k in range(n):
    if y_hiraki[k] > treshold or y_hiraki_crypto[k] > treshold :
        kth_row = log.iloc[k,2:8]  
        print("========================")
        print("La log suivante a été détectée positive:")
        print(kth_row)
        print(f"Model Brutforce-BrutforceXML-Probing avec une proba: {y_hiraki[k]}")
        print(f"Model XMRIGCC CryptoMiner avec une proba: {y_hiraki_crypto[k]}")
        print(f"Model FTP/SSH-Bruteforce  avec une proba: {y_cicids[k]}")