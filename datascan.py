'''
Scans files in dataset directory for parameters when needed for debugging
'''

import csv

filename = "dataset/totalData"
attacks = [] # list of attacks

for i in range(4):
    filename = "dataset/totalData"
    filename = filename + str(i+1) + ".csv"
    print("SCANNING FILE: ", filename)
    with open(filename,"r") as file:
        data = list(csv.reader(file))
        for entry in data:
           # print(entry)
            if entry[9] not in attacks:
                attacks.append(entry[9])


print(attacks)