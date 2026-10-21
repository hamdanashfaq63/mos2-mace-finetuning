"""Split the labeled data into train / validation / test sets.

The biaxial strain series is kept out of training entirely and used later
as an equation-of-state test. The other 110 structures are shuffled:
20 for testing, 10 for validation and 80 for training. The first 20 and 40
training structures are also saved separately for a learning curve, and the
training set is saved again with the vacancy structures added.
"""
from glob import glob
import numpy as np
from ase.io import read, write

data = [a for f in sorted(glob("data/dft_part*.xyz")) for a in read(f, ":")]
write("data/dft_labeled.xyz", data)
eos = sorted((a for a in data if a.info["source"] == "strain"), key=lambda a: a.info["strain"])
rest = [a for a in data if a.info["source"] != "strain"]

idx = np.random.default_rng(42).permutation(len(rest))
test = [rest[i] for i in idx[:20]]
valid = [rest[i] for i in idx[20:30]]
train = [rest[i] for i in idx[30:]]

# smaller nested training sets for the learning curve
subsets = [(f"train_{n}", train[:n]) for n in (20, 40)]
# the training set plus the 26 vacancy structures from 03_vacancy_data.py
subsets.append(("train_plus_vacancy", train + read("data/vacancy_train.xyz", ":")))

for name, part in [("train", train), ("valid", valid), ("test", test), ("eos", eos)] + subsets:
    write(f"data/{name}.xyz", part)
    print(f"{name:5s}: {len(part)} structures")
