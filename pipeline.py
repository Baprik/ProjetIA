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


def capture_network_traffic(capture_file,duration = 60):
    # Définir le nom du fichier de capture avec le format de la date et de l'heure actuelle
    
    
    # Fonction pour enregistrer les paquets capturés dans un fichier PCAP
    def save_capture(pkt):
        wrpcap(capture_file, pkt, append=True)
        print(f"Packet captured and saved to {capture_file}")


    start_time = time.time()
    print("Starting scanning network")
    while time.time() - start_time < duration:
        sniff(prn=save_capture, timeout=60)
    print("Ending scanning network")


parser = argparse.ArgumentParser(description="Description du script")
parser.add_argument('--t', type=int, default=60, help='Duration of the scanning network, a multiple of 60 is required')


args = parser.parse_args()

print('Duration of scanning network: ', args.t)

file = "captures/"
capture_file = f"network_capture_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pcap"
capture_network_traffic(file + capture_file, duration=  args.t)


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
X, ids = loadCapCicids(path)
model_Cicids = MalwareDetector4(X)
state_dict = torch.load("model/Cicids/MalwareDectetor4Ccids.pt")
model_Cicids.load_state_dict(state_dict)
model_Cicids.eval()
y_cicids = model_Cicids(X)
n = len(y_cicids)
"""



treshold = 0.01
for k in range(n):
    if y_hiraki[k] > treshold or y_hiraki_crypto[k] > treshold:
        kth_row = log.iloc[k,2:8]  
        print("========================")
        print("La log suivante a été détectée positive:")
        print(kth_row)
        print(f"Model Brutforce-BrutforceXML-Probing avec une proba: {y_hiraki[k]}")
        print(f"Model XMRIGCC CryptoMiner avec une proba: {y_hiraki_crypto[k]}")