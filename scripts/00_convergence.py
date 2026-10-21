"""Compare the DFT settings (400 eV, 4x4x1 k-points) with tighter ones on a rattled cell."""
import json
import numpy as np
from ase.build import mx2
from dft import label

atoms = mx2("MoS2", "2H", a=3.18, thickness=3.19, vacuum=7.0).repeat((2, 2, 1))
atoms.pbc = (True, True, False)
atoms.positions += np.random.default_rng(7).normal(0, 0.05, atoms.positions.shape)

ref = label(atoms, ecut=600, kpts=(6, 6, 1))


def compare(result):
    dE = 1000 * (result.info["REF_energy"] - ref.info["REF_energy"]) / len(atoms)
    dF = 1000 * np.abs(result.arrays["REF_forces"] - ref.arrays["REF_forces"]).max()
    return {"dE_meV_per_atom": round(float(dE), 2), "max_dF_meV_per_A": round(float(dF), 2)}


report = {"reference": "600 eV, 6x6x1", "cutoff_eV (k=6x6x1)": {}, "kpoints (600 eV)": {}}
for ecut in (300, 400, 500):
    report["cutoff_eV (k=6x6x1)"][ecut] = compare(label(atoms, ecut=ecut, kpts=(6, 6, 1)))
    print(ecut, report["cutoff_eV (k=6x6x1)"][ecut], flush=True)
for k in (3, 4, 5):
    report["kpoints (600 eV)"][f"{k}x{k}x1"] = compare(label(atoms, ecut=600, kpts=(k, k, 1)))
    print(k, report["kpoints (600 eV)"][f"{k}x{k}x1"], flush=True)
report["production (400 eV, 4x4x1)"] = compare(label(atoms, ecut=400, kpts=(4, 4, 1)))

json.dump(report, open("results/convergence.json", "w"), indent=2)
print(json.dumps(report, indent=2))
