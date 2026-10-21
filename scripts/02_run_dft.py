"""Label every structure with PBE DFT (GPAW, plane waves; settings in dft.py).

GPAW runs on one core here, so the structures can be split across several
processes:  python scripts/02_run_dft.py 0 2  and  python scripts/02_run_dft.py 1 2
Each process appends its results to data/dft_part<k>.xyz one structure at a time,
so a run can be stopped and restarted without losing work.
"""
import os
import sys
import time
from ase.io import read, write
from dft import label

part, nparts = (int(x) for x in sys.argv[1:3]) if len(sys.argv) > 2 else (0, 1)
out_file = f"data/dft_part{part}.xyz"

structures = read("data/structures.xyz", ":")[part::nparts]
done = len(read(out_file, ":")) if os.path.exists(out_file) else 0
print(f"part {part}: {done}/{len(structures)} already labeled", flush=True)

for i, atoms in enumerate(structures[done:], start=done):
    t0 = time.time()
    labeled = label(atoms)
    write(out_file, labeled, append=True)
    print(f"[{i + 1}/{len(structures)}] {atoms.info['source']:8s} "
          f"E = {labeled.info['REF_energy']:.4f} eV  "
          f"max|F| = {abs(labeled.arrays['REF_forces']).max():.3f} eV/A  ({time.time() - t0:.0f} s)",
          flush=True)
