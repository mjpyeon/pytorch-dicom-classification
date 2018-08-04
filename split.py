import numpy as np
import glob
import shutil
import sys
import os

def split_and_save(root_dir, k):
    fnames = glob.glob(os.path.join(root_dir, "*"))
    np.random.shuffle(fnames)
    idx = np.array_split(np.arange(len(fnames)), k)
    splitted = [[fnames[j] for j in idx[i]] for i in range(k)]
    for i, arr in enumerate(splitted):
        dest = os.path.join(root_dir, "..", "%s-%d"%(root_dir.split('/')[-1], i))
        try:
            os.mkdir(dest)
        except:
            pass
        for fname in arr:
            shutil.copy(fname, os.path.join(dest, fname.split('/')[-1]))

split_and_save(sys.argv[1], int(sys.argv[2]))
