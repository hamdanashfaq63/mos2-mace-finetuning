"""Build monolayer MoS2 configurations to label with DFT.

Three kinds of structures, all 2x2 supercells (12 atoms):
  strain  - unrattled cells under biaxial strain from -4% to +4% (held out as an equation-of-state test)
  rattle  - random strain (+-3%) plus random atomic displacements
  md      - snapshots from short molecular dynamics runs with the MACE-MP-0 foundation model at 300, 600 and 900 K
"""
import numpy as np
from ase.build import mx2
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import mace_mp

A0 = 3.18          # in-plane lattice constant (Angstrom), close to the PBE value
rng = np.random.default_rng(0)


def base_cell(a=A0):
    atoms = mx2("MoS2", "2H", a=a, thickness=3.19, vacuum=7.0).repeat((2, 2, 1))
    atoms.pbc = (True, True, False)
    return atoms


def strained(atoms, exx, eyy):
    atoms = atoms.copy()
    cell = atoms.cell.array.copy()
    cell[0] *= 1 + exx
    cell[1] *= 1 + eyy
    atoms.set_cell(cell, scale_atoms=True)
    return atoms


structures = []

# 1) biaxial strain series
for eps in np.linspace(-0.04, 0.04, 9):
    s = strained(base_cell(), eps, eps)
    s.info["source"] = "strain"
    s.info["strain"] = float(eps)
    structures.append(s)

# 2) random strain + rattle
for i in range(50):
    s = strained(base_cell(), *rng.uniform(-0.03, 0.03, 2))
    s.positions += rng.normal(0, rng.uniform(0.02, 0.10), s.positions.shape)
    s.info["source"] = "rattle"
    structures.append(s)

# 3) MD snapshots with the foundation model
calc = mace_mp(model="small", default_dtype="float32", device="cpu")
for T in (300, 600, 900):
    atoms = base_cell()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=T, friction=0.02, rng=rng)
    dyn.run(500)                       # 0.5 ps equilibration
    for _ in range(20):                # 20 snapshots, 50 fs apart
        dyn.run(50)
        s = atoms.copy()
        s.calc = None
        s.info["source"] = f"md_{T}K"
        structures.append(s)

write("data/structures.xyz", structures)
print(f"wrote {len(structures)} structures")
