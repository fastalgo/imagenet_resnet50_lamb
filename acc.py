import sys
import numpy as np

print 'Usage: python acc.py Filename'
filename = sys.argv[1];
config = ""
best_acc = 100.0
with open(filename) as f:
    for line in f:
        if "learning_rate " in line:
            print(line)
        if "label_smoothing" in line:
            print(line)
        if "weight_decay" in line:
            print(line)
        if "warmup" in line:
            print(line)
        if "top_1_accuracy" in line:
            print(line)
        if "beta1 is" in line:
            print(line)
        if "beta2 is" in line:
            print(line)
        if "eps is" in line:
            print(line)
            #mystr = line.split();
            #acclist = mystr[-1].split('%');
            #acc = float(acclist[0]);
            #if acc < best_acc:
                #best_acc = acc
                #print(config)
                #print("lowest error rate: " + str(best_acc))
