"""Build and label MoS2 structures with a sulfur vacancy.

Sulfur vacancies are the most common point defect in MoS2. The main training
set has none, so this script makes two separate sets:
  data/vacancy_test.xyz   12 structures, used only to test transfer to the defect
  data/vacancy_train.xyz  26 structures, added to training in a second
                          fine-tuning run to see how much defect data fixes it
Each set is skipped if its file already exists.
"""
import os
import numpy as np
from ase.build import mx2
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import mace_mp
from dft import label

calc = mace_mp(model="small", default_dtype="float32", device="cpu")


def vacancy_cell():
    atoms = mx2("MoS2", "2H", a=3.18, thickness=3.19, vacuum=7.0).repeat((2, 2, 1))
    atoms.pbc = (True, True, False)
    del atoms[[a.index for a in atoms if a.symbol == "S"][0]]     # remove one S atom
    return atoms


def make_set(seed, n_rattle, temperatures, n_per_T, gap):
    rng = np.random.default_rng(seed)
    structures = []
    for _ in range(n_rattle):                                   # rattled
        s = vacancy_cell()
        s.positions += rng.normal(0, 0.05, s.positions.shape)
        structures.append(s)
    for T in temperatures:                                      # MD snapshots from the foundation model
        atoms = vacancy_cell()
        atoms.calc = calc
        MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
        dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=T, friction=0.02, rng=rng)
        dyn.run(500)
        for _ in range(n_per_T):
            dyn.run(gap)
            structures.append(atoms.copy())
    labeled = []
    for i, s in enumerate(structures):
        s.calc = None
        s.info["source"] = "S_vacancy"
        labeled.append(label(s))
        print(f"[{i + 1}/{len(structures)}] done", flush=True)
    return labeled


if not os.path.exists("data/vacancy_test.xyz"):
    write("data/vacancy_test.xyz", make_set(seed=3, n_rattle=4, temperatures=(300, 600), n_per_T=4, gap=100))
if not os.path.exists("data/vacancy_train.xyz"):
    write("data/vacancy_train.xyz", make_set(seed=11, n_rattle=8, temperatures=(300, 600, 900), n_per_T=6, gap=80))
