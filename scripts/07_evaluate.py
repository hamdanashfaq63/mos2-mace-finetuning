"""Evaluate every model against held-out DFT data.

Models: the MACE-MP-0 foundation model with no extra training (zero-shot),
MACE trained from scratch and MACE-MP-0 fine-tuned (each on 20, 40 and 80 structures),
and MACE-MP-0 fine-tuned on the 80 structures plus 26 sulfur-vacancy structures.

Tests:
  test set        - 20 held-out structures like the training data (energies and forces)
  S-vacancy set   - 12 structures with a sulfur vacancy (forces only). Only the last
                    model saw vacancies in training, and never these 12.
  strain series   - 9 unrattled strained cells, used to predict the lattice constant

Energy reference: DFT codes measure energy from different zero points, so each
model's energies are shifted by one constant per atom, fitted on the training
set. Every structure in the train and test sets has the same composition (Mo4S8),
so one constant is enough. The vacancy cells have a different composition, so
only their forces are compared.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from mace.calculators import mace_mp, MACECalculator

train, test, eos, vac = (read(f"data/{n}.xyz", ":") for n in ("train", "test", "eos", "vacancy_test"))


def load(path):
    return MACECalculator(model_paths=path, default_dtype="float64", device="cpu")


models = {
    "MACE-MP-0 zero-shot": mace_mp(model="small", default_dtype="float64", device="cpu"),
    "From scratch (20)": load("models/mos2_from_scratch_20.model"),
    "From scratch (40)": load("models/mos2_from_scratch_40.model"),
    "From scratch (80)": load("models/mos2_from_scratch.model"),
    "Fine-tuned (20)": load("models/mos2_finetuned_20.model"),
    "Fine-tuned (40)": load("models/mos2_finetuned_40.model"),
    "Fine-tuned (80)": load("models/mos2_finetuned.model"),
    "Fine-tuned (80) + 26 vacancy": load("models/mos2_finetuned_vacancy.model"),
}


def predict(calc, structures):
    e, f = [], []
    for atoms in structures:
        a = atoms.copy()
        a.calc = calc
        e.append(a.get_potential_energy() / len(a))
        f.append(a.get_forces())
    return np.array(e), np.concatenate(f)


def reference(structures):
    return (np.array([a.info["REF_energy"] / len(a) for a in structures]),
            np.concatenate([a.arrays["REF_forces"] for a in structures]))


def lattice_constant(a, e):
    fit = np.polyfit(a, e, 3)
    roots = np.roots(np.polyder(fit))
    return float([r.real for r in roots if abs(r.imag) < 1e-9 and np.polyval(np.polyder(fit, 2), r.real) > 0][0])


mae = lambda x, y: float(1000 * np.mean(np.abs(x - y)))
rmse = lambda x, y: float(1000 * np.sqrt(np.mean((x - y) ** 2)))

e_train, _ = reference(train)
e_test, f_test = reference(test)
_, f_vac = reference(vac)
e_eos, _ = reference(eos)
a_lat = np.array([3.18 * (1 + a.info["strain"]) for a in eos])

metrics = {"DFT (PBE)": {"lattice_constant_A": lattice_constant(a_lat, e_eos)}}
preds = {}
for name, calc in models.items():
    shift = np.mean(e_train - predict(calc, train)[0])
    e_pred, f_pred = predict(calc, test)
    _, f_vac_pred = predict(calc, vac)
    e_eos_pred = predict(calc, eos)[0] + shift
    metrics[name] = {
        "test_energy_MAE_meV_per_atom": mae(e_pred + shift, e_test),
        "test_force_MAE_meV_per_A": mae(f_pred, f_test),
        "test_force_RMSE_meV_per_A": rmse(f_pred, f_test),
        "vacancy_force_MAE_meV_per_A": mae(f_vac_pred, f_vac),
        "lattice_constant_A": lattice_constant(a_lat, e_eos_pred),
        # shape of the energy-strain curve, after removing any constant offset
        "strain_curve_RMSE_meV_per_atom": rmse(e_eos_pred - np.mean(e_eos_pred - e_eos), e_eos),
    }
    preds[name] = (f_pred, e_eos_pred)
    print(name, json.dumps(metrics[name]), flush=True)

json.dump(metrics, open("results/metrics.json", "w"), indent=2)

# force parity plots for the three main models
main = ["MACE-MP-0 zero-shot", "From scratch (80)", "Fine-tuned (80)"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
lim = 1.05 * np.abs(f_test).max()
for ax, name in zip(axes, main):
    ax.scatter(f_test.ravel(), preds[name][0].ravel(), s=5, alpha=0.4)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="DFT force (eV/Å)", ylabel="Model force (eV/Å)",
           title=f"{name}\nforce MAE = {metrics[name]['test_force_MAE_meV_per_A']:.0f} meV/Å")
    ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("results/force_parity.png", dpi=200, bbox_inches="tight")

# learning curve
n = [20, 40, 80]
plt.figure(figsize=(5.2, 4))
plt.plot(n, [metrics[f"Fine-tuned ({k})"]["test_force_MAE_meV_per_A"] for k in n], "o-", label="Fine-tuned MACE-MP-0")
plt.plot(n, [metrics[f"From scratch ({k})"]["test_force_MAE_meV_per_A"] for k in n], "s-", label="Trained from scratch")
plt.axhline(metrics["MACE-MP-0 zero-shot"]["test_force_MAE_meV_per_A"], color="0.5", ls="--", label="MACE-MP-0 zero-shot")
plt.yscale("log")
plt.xticks(n)
plt.xlabel("Training structures")
plt.ylabel("Test force MAE (meV/Å, log scale)")
plt.title("Learning curve")
plt.legend()
plt.tight_layout()
plt.savefig("results/learning_curve.png", dpi=200, bbox_inches="tight")

# vacancy test
names = list(models)
plt.figure(figsize=(6.4, 4))
plt.barh(names, [metrics[k]["vacancy_force_MAE_meV_per_A"] for k in names])
plt.gca().invert_yaxis()
plt.xlabel("Force MAE on S-vacancy test set (meV/Å)")
plt.title("Transfer to an unseen defect")
plt.tight_layout()
plt.savefig("results/vacancy_test.png", dpi=200, bbox_inches="tight")

# strain test
plt.figure(figsize=(5.2, 4))
plt.plot(a_lat, 1000 * (e_eos - e_eos.min()), "ko", label="DFT (PBE)")
for name in main:
    plt.plot(a_lat, 1000 * (preds[name][1] - preds[name][1].min()), "-", label=name)
plt.xlabel("In-plane lattice constant (Å)")
plt.ylabel("Energy relative to minimum (meV/atom)")
plt.title("Held-out biaxial strain test")
plt.legend()
plt.tight_layout()
plt.savefig("results/strain_test.png", dpi=200, bbox_inches="tight")
