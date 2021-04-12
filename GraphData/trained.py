import random
import sys
import os

if len(sys.argv) != 2:
    os._exit(1)

hideP = 15

suffix = ".txt"
filename = sys.argv[1].split(".")[-2]
filepath = filename + "-t" + suffix

fin = open(sys.argv[1], "r") # sys.argv: trained.py grqc.txt
fout = open(filepath, "w")

#lineSt = fin.readline().strip().split()
#N = int(lineSt[0])
#E = int(lineSt[1])
for line in fin.readlines():
    p = random.randint(1, 100)
    if p <= hideP:
        continue
    lineW = line.strip().split()
    x = int(lineW[0])-1
    y = int(lineW[1])-1
    #if x == y:
    #    continue
    z = lineW[2]
    fout.write(str(x) + " " + str(y) + " " + str(z) + "\n")

fout.close()
fin.close()
