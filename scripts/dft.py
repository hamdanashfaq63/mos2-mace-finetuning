"""DFT settings shared by every script: PBE, plane waves, GPAW."""
import gc
from gpaw import GPAW, PW, FermiDirac


def label(atoms, ecut=400, kpts=(4, 4, 1)):
    """Return a copy of `atoms` with DFT energy and forces stored as REF_energy / REF_forces."""
    atoms = atoms.copy()
    atoms.calc = GPAW(mode=PW(ecut), xc="PBE", kpts=kpts, occupations=FermiDirac(0.05), txt=None)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms.calc = None                   # free GPAW's memory before the next calculation
    gc.collect()
    atoms.info["REF_energy"] = energy
    atoms.arrays["REF_forces"] = forces
    return atoms
