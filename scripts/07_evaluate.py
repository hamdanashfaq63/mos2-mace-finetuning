"""Evaluate all models on held-out DFT data and make the figures.

Test set: the held-out 600 K MD run and 10 rattled cells (energies and forces).
Vacancy set: 12 cells with a sulfur vacancy (forces only, since their
composition differs). Strain series: 9 unrelaxed strained cells.

GPAW and MACE-MP-0 use different energy zero points, so each model's energies
are shifted by one constant per atom fitted on the training set.
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
    "From scratch (70)": load("models/mos2_from_scratch.model"),
    "From scratch (70) seed 2": load("models/mos2_from_scratch_seed2.model"),
    "From scratch (70) seed 3": load("models/mos2_from_scratch_seed3.model"),
    "Fine-tuned (20)": load("models/mos2_finetuned_20.model"),
    "Fine-tuned (40)": load("models/mos2_finetuned_40.model"),
    "Fine-tuned (70)": load("models/mos2_finetuned.model"),
    "Fine-tuned (70) seed 2": load("models/mos2_finetuned_seed2.model"),
    "Fine-tuned (70) seed 3": load("models/mos2_finetuned_seed3.model"),
    "Fine-tuned (70) + 26 vacancy": load("models/mos2_finetuned_vacancy.model"),
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

# mean and standard deviation over the three seeds at 70 structures
for label in ("From scratch", "Fine-tuned"):
    runs = [metrics[f"{label} (70)"]] + [metrics[f"{label} (70) seed {k}"] for k in (2, 3)]
    metrics[f"{label} (70), 3 seeds"] = {key: {"mean": float(np.mean([r[key] for r in runs])),
                                               "std": float(np.std([r[key] for r in runs]))} for key in runs[0]}

json.dump(metrics, open("results/metrics.json", "w"), indent=2)

# force parity for the three main models, plus the distribution of their errors
main = ["MACE-MP-0 zero-shot", "From scratch (70)", "Fine-tuned (70)"]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
lim = 1.05 * np.abs(f_test).max()
for ax, name in zip(axes, main):
    ax.scatter(f_test.ravel(), preds[name][0].ravel(), s=5, alpha=0.4)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="DFT force (eV/Å)", ylabel="Model force (eV/Å)",
           title=f"{name}\nforce MAE = {metrics[name]['test_force_MAE_meV_per_A']:.0f} meV/Å")
    ax.set_aspect("equal")
bins = np.logspace(-1, 3.5, 30)
for name in main:
    axes[3].hist(1000 * np.abs(preds[name][0] - f_test).ravel(), bins=bins, histtype="step", lw=1.5, label=name)
axes[3].set(xscale="log", xlabel="|Force error| (meV/Å)", ylabel="Force components", title="Error distribution")
axes[3].legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/force_parity.png", dpi=200, bbox_inches="tight")

# learning curve: test set (left) and unseen S-vacancy set (right)
n = [20, 40, 70]
zero = metrics["MACE-MP-0 zero-shot"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, key, title in zip(axes, ["test_force_MAE_meV_per_A", "vacancy_force_MAE_meV_per_A"],
                          ["Pristine test set (held-out 600 K run + rattled)", "S-vacancy test set"]):
    curves = {}
    for label, marker in [("Fine-tuned", "o"), ("From scratch", "s")]:
        y = [metrics[f"{label} ({k})"][key] for k in n[:2]] + [metrics[f"{label} (70), 3 seeds"][key]["mean"]]
        err = [0, 0, metrics[f"{label} (70), 3 seeds"][key]["std"]]
        ax.errorbar(n, y, yerr=err, marker=marker, capsize=4,
                    label=f"{label} MACE" if label == "From scratch" else "Fine-tuned MACE-MP-0")
        curves[label] = y
    # label the higher point above and the lower point below, beside the error bar
    fmt = "{:.1f}" if key.startswith("test") else "{:.0f}"
    a, b = curves.values()
    for i, x in enumerate(n):
        for v, w in ((a[i], b[i]), (b[i], a[i])):
            dx, ha = {0: (-8, "right"), 1: (0, "center"), 2: (-10, "right")}[i]
            dy = (-3 if i == 0 else 7 if v >= w else -13)
            ax.annotate(fmt.format(v), (x, v), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8)
    ax.set(xticks=n, xlim=(6, 74), xlabel="Training structures", ylabel="Force MAE (meV/Å)", title=title)
    ax.set_ylim(bottom=0)
axes[0].text(0.97, 0.35, f"MACE-MP-0 with no fine-tuning: {zero['test_force_MAE_meV_per_A']:.0f} meV/Å (off scale)",
             transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8, color="0.35")
axes[1].axhline(zero["vacancy_force_MAE_meV_per_A"], color="0.5", ls="--", label="MACE-MP-0, no fine-tuning")
axes[1].set_ylim(0, 1.1 * max(axes[1].get_ylim()[1], zero["vacancy_force_MAE_meV_per_A"]))
axes[0].legend(loc="lower left")
axes[1].legend(loc="lower left")
fig.tight_layout()
fig.savefig("results/learning_curve.png", dpi=200, bbox_inches="tight")

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
