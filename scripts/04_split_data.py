"""Split the pristine data: strain series held out, then 80 train, 10 valid, 20 test.

Also writes the 20 and 40 structure subsets for the learning curve and the
training set with the vacancy structures added.
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

subsets = [(f"train_{n}", train[:n]) for n in (20, 40)]
subsets.append(("train_plus_vacancy", train + read("data/vacancy_train.xyz", ":")))

for name, part in [("train", train), ("valid", valid), ("test", test), ("eos", eos)] + subsets:
    write(f"data/{name}.xyz", part)
    print(f"{name:5s}: {len(part)} structures")
