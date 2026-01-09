
import torch
import glob
from natsort import natsorted
import pandas as pd
import numpy as np

ds = []
for i in [0, 1, 2, 3, 4]:
    d = {}
    files = natsorted(glob.glob('fits/model=Llama-3.1-Centaur-70B_layer=' + str(i) + '_roi=*'))
    for file in files:
        r2_scores = torch.load(file)
        d[file.removeprefix('fits/model=Llama-3.1-Centaur-70B_layer=' + str(i) + '_roi=').removesuffix('.pth')] = r2_scores[:, :, :, 1].mean()
    print(len(d))
    ds.append(d)
df = pd.DataFrame(ds)
df.to_csv('../results/feher2023rethinking/tst_centaur_alignment.csv', index=False)
twostep_centaur = df.values.mean(1)
print(twostep_centaur)

ds = []
for i in [0, 1, 2, 3, 4]:
    d = {}
    files = natsorted(glob.glob('fits/model=Meta-Llama-3.1-70B_layer=' + str(i) + '_roi=*'))
    for file in files:
        r2_scores = torch.load(file)
        d[file.removeprefix('fits/model=Meta-Llama-3.1-70B_layer=' + str(i) + '_roi=').removesuffix('.pth')] = r2_scores[:, :, :, 1].mean()
    print(len(d))
    ds.append(d)
df = pd.DataFrame(ds)
df.to_csv('../results/feher2023rethinking/tst_llama_alignment.csv', index=False)
twostep_llama = df.values.mean(1)
print(twostep_llama)
