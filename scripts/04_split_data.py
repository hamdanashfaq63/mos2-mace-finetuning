"""Split the pristine data so that no MD trajectory is shared between train and test.

Neighboring MD snapshots are correlated, so the whole 600 K trajectory (20 frames)
goes to the test set, together with 10 independent rattled cells. 10 more rattled
cells are used for validation. Training gets the 300 K and 900 K trajectories and
the remaining 30 rattled cells (70 structures). The strain series is held out.
Also writes 20 and 40 structure subsets and the training set plus vacancy data.
"""
from glob import glob
import numpy as np
from ase.io import read, write

data = [a for f in sorted(glob("data/dft_part*.xyz")) for a in read(f, ":")]
write("data/dft_labeled.xyz", data)
by_source = lambda s: [a for a in data if a.info["source"] == s]

eos = sorted(by_source("strain"), key=lambda a: a.info["strain"])
rattle = by_source("rattle")
rng = np.random.default_rng(42)
idx = rng.permutation(len(rattle))

test = by_source("md_600K") + [rattle[i] for i in idx[:10]]
valid = [rattle[i] for i in idx[10:20]]
train = [rattle[i] for i in idx[20:]] + by_source("md_300K") + by_source("md_900K")
train = [train[i] for i in rng.permutation(len(train))]

subsets = [(f"train_{n}", train[:n]) for n in (20, 40)]
subsets.append(("train_plus_vacancy", train + read("data/vacancy_train.xyz", ":")))

for name, part in [("train", train), ("valid", valid), ("test", test), ("eos", eos)] + subsets:
    write(f"data/{name}.xyz", part)
    print(f"{name:5s}: {len(part)} structures")
