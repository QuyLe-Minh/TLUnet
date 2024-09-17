import csv
import json
import os
import pandas as pd


path = "C:/Users/Admin/OneDrive - hcmut.edu.vn/A.I. references/ComVis/Research/Results/unetrpp"
json_path = os.path.join(path, 'summary.json')

organs_ids = ['1', '2', '3', '4', '6', '7', '8', '11']
organs_names = ['Spl', 'RKid', 'LKid', "Gal", "Liv", "Sto", "Aor", "Pan"]

with open(json_path, 'r') as f:
    data = json.load(f)
    
all = data['results']['all']


with open('unetrpp.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Case'] + organs_names)
    for d in all:
        patient = d['reference'].split('/')[-1].split('.')[0]
        row = [patient]
        for organ_id in organs_ids:
            row.append(d[organ_id]["Dice"])
        writer.writerow(row)
        
# df = pd.read_csv('output.csv')
# df.to_excel("output.xlsx", index=False)