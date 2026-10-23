"""MD with the fine-tuned model on a 6x6 cell (108 atoms).

10 ps at 300 K with a Langevin thermostat, then 2 ps NVE to check energy
conservation. The shortest Mo-S distance is tracked to catch structural failure.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.build import mx2
from ase import units
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator

atoms = mx2("MoS2", "2H", a=3.18, thickness=3.19, vacuum=7.0).repeat((6, 6, 1))
atoms.pbc = (True, True, False)
atoms.calc = MACECalculator(model_paths="models/mos2_finetuned.model", default_dtype="float64", device="cpu")
MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))

mo = [a.index for a in atoms if a.symbol == "Mo"]
s = [a.index for a in atoms if a.symbol == "S"]
log = {"time_ps": [], "T_K": [], "E_total_eV": [], "min_MoS_A": [], "phase": []}


def record(dyn, phase):
    d = atoms.get_all_distances(mic=True)[np.ix_(mo, s)]
    log["time_ps"].append(len(log["time_ps"]) * 0.01)
    log["T_K"].append(atoms.get_temperature())
    log["E_total_eV"].append(atoms.get_total_energy())
    log["min_MoS_A"].append(float(d.min()))
    log["phase"].append(phase)


nvt = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01, rng=np.random.default_rng(1))
nvt.attach(lambda: record(nvt, "NVT"), interval=10)
nvt.run(10000)                                  # 10 ps

nve = VelocityVerlet(atoms, 1.0 * units.fs)
nve.attach(lambda: record(nve, "NVE"), interval=10)
nve.run(2000)                                   # 2 ps

t = np.array(log["time_ps"])
nvt_mask = np.array(log["phase"]) == "NVT"
E_nve = np.array(log["E_total_eV"])[~nvt_mask]
summary = {
    "atoms": len(atoms),
    "mean_T_last_5ps_K": float(np.mean(np.array(log["T_K"])[nvt_mask][-500:])),
    "min_MoS_distance_A": float(min(log["min_MoS_A"])),
    "NVE_energy_drift_meV_per_atom_per_ps": float(1000 * (E_nve[-1] - E_nve[0]) / len(atoms) / 2.0),
    "NVE_energy_std_meV_per_atom": float(1000 * np.std(E_nve) / len(atoms)),
}
json.dump({"summary": summary, "log": log}, open("results/md_log.json", "w"))
print(json.dumps(summary, indent=2))

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
fig, ax = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
ax[0].plot(t, log["T_K"], lw=0.8, color="#0072B2")
ax[0].axhline(300, color="k", ls="--", lw=1)
ax[0].set_ylabel("Temperature (K)")
ax[1].plot(t, log["min_MoS_A"], lw=0.8, color="#0072B2")
ax[1].set_ylabel("Shortest Mo-S distance (Å)")
ax[2].plot(t, 1000 * (np.array(log["E_total_eV"]) - log["E_total_eV"][0]) / len(atoms), lw=0.8, color="#0072B2")
ax[2].set_ylabel("Total energy (meV/atom)")
ax[2].set_xlabel("Time (ps)")
for a in ax:
    a.axvspan(10, t[-1], color="0.9", zorder=0)
ax[0].text(11, 0.95, "NVE", transform=ax[0].get_xaxis_transform(), ha="center", va="top", fontsize=8, color="0.35")
ax[0].set_title("Fine-tuned model, 108-atom MoS$_2$: 10 ps at 300 K (Langevin), then 2 ps NVE")
fig.tight_layout()
fig.savefig("results/md_stability.png", dpi=200, bbox_inches="tight")
