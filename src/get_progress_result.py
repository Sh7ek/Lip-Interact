import numpy as np
import os

with open("../resource/subjects.txt") as f:
    subjects = f.readlines()
    subjects = [x.strip() for x in subjects]

g = 9

X = np.zeros(shape=(len(subjects), 5), dtype=np.float32)
for i, s in enumerate(subjects):
    for p in [1, 2, 3, 4, 5]:
        fn = "../resource/progress/group_" + str(g) + "_" + s + "_" + str(p) + ".txt"
        if os.path.exists(fn):
            lines = []
            with open(fn) as fo:
                lines = fo.readlines()
                lines = [float(x.strip()) for x in lines]
            a = np.mean(lines)
            if a == 0:
                print("error")
            X[i][p-1] = a * 100
        else:
            print("file not exist: {}".format(fn))

fw = open("../resource/progress_group_" + str(g) + ".txt", 'w')
for l in X:
    for a in l:
        fw.write(str(a) + "\t")
    fw.write("\n")
fw.close()
