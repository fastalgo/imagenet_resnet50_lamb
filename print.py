import sys
import numpy as np
loss_list = []
step_list = []
filename = sys.argv[1];
max_top1 = 0.0
max_top5 = 0.0
config1 = ''
config2 = ''
config3 = ''
with open(filename) as f:
    for line in f:
        if "learning" in line:
            config1 = line
        if "warmup" in line:
            config2 = line
        if "weight_decay" in line:
            config3 = line
        if "Saving dict" in line:
            #print(line)
            a = line.replace(',', '')
            #b = a.replace(']', ' ')
            #print(b)
            mystr = a.split();
            top1 = float(mystr[-1])
            top5 = float(mystr[-4])
            if top1 > max_top1:
                print(config1)
                print(config2)
                print(config3)
                print(line)
                max_top1 = top1
            if top5 > max_top5:
                print(config1)
                print(config2)
                print(config3)
                print(line)
                max_top5 = top5
    #print(loss_list)
    #print(step_list)
