import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from newtonnet.utils.ase_interface import MLAseCalculator
from torch_geometric.data import Data
from ase.data import atomic_numbers
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ===========================
# parse_xyz (from your version)
# ===========================
def parse_xyz(raw_path: str, precision=torch.float64):
    data_list = []
    with open(raw_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].strip().isdigit():
            i += 1
            continue
        natoms = int(lines[i].strip())
        i += 1
        header = lines[i].strip()
        i += 1

        # --- Energy ---
        energy = None
        if "energy=" in header:
            try:
                energy = float(header.split("energy=")[1].split()[0])
            except Exception:
                pass

        # --- Dipole ---
        dipole = None
        if "dipole=" in header:
            try:
                dip_str = header.split("dipole=")[1].split()[0]
                dipole = torch.tensor([float(x) for x in dip_str.replace(",", " ").split()],
                                      dtype=precision)
            except Exception:
                pass

        # --- Atomic block ---
        atom_lines = lines[i:i+natoms]
        i += natoms
        symbols, pos, forces = [], [], []
        for line in atom_lines:
            parts = line.split()
            if len(parts) >= 7:
                symbols.append(parts[0])
                pos.append([float(x) for x in parts[1:4]])
                forces.append([float(x) for x in parts[4:7]])

        z = torch.tensor([atomic_numbers[s] for s in symbols], dtype=torch.int)
        pos = torch.tensor(pos, dtype=precision)
        forces = torch.tensor(forces, dtype=precision)

        data = Data()
        data.z = z
        data.pos = pos
        data.force = forces
        if energy is not None:
            data.energy = torch.tensor([energy], dtype=precision)
        if dipole is not None:
            data.dipole = dipole.reshape(1, 3)

        data_list.append(data)
    return data_list


# ===========================
# Helper for hexbin plotting
# ===========================
def hexbin_on_ax(ax, x, y, xlabel, ylabel, title, panel_letter, scale=1.0, manual_axis=False):
    """??? ax ??? hexbin + ?? + ??"""
    x, y = _prep_xy(x, y, scale=scale, manual_axis=manual_axis)

    # ?????
    slope, intercept, r_value, _, _ = linregress(x, y)
    mae = np.mean(np.abs(y - x))
    rmse = np.sqrt(np.mean((y - x) ** 2))

    # ????????
    lo, hi = _equal_limits(x, y)
    hb = ax.hexbin(x, y, gridsize=60, cmap='viridis', bins='log')
    ax.set_xlim(lo, hi);
    ax.set_ylim(lo, hi)
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, slope * xs + intercept, 'r--', linewidth=1)
    ax.set_aspect('equal', adjustable='box')

    # ?????????
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.text(0.18, 0.96, f"RMSE = {rmse:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, va='top', ha='left')
    ax.text(0.02, 0.98, panel_letter, transform=ax.transAxes,
            va='top', ha='left', fontsize=12, fontweight='bold')

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(hb, cax=cax)
    cbar.set_label('log$_{10}$(count)')

def _prep_xy(x, y, scale=1.0, manual_axis=False):
    """??? -> ?? -> ?? NaN/Inf -> ??????"""
    x = np.asarray(x, dtype=float).copy() * scale
    y = np.asarray(y, dtype=float).copy() * scale
    x = x.ravel();
    y = y.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask];
    y = y[mask]
    if manual_axis:
        ref = x.mean()
        x = x - ref
        y = y - ref
    return x, y


def _equal_limits(x, y, pad_ratio=0.05):
    data_min = min(x.min(), y.min())
    data_max = max(x.max(), y.max())
    span = data_max - data_min
    pad = pad_ratio * span if span > 0 else 1.0
    lo, hi = data_min - pad, data_max + pad
    return lo, hi

# ===========================
# Evaluation
# ===========================
def evaluate(model_path, test_xyz, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Loading model from {model_path}")
    print(f"📄 Reading test XYZ from {test_xyz}")

    data_list = parse_xyz(test_xyz)
    print(f"✅ Parsed {len(data_list)} structures")

    energies_ref = np.array([d.energy.item() for d in data_list if hasattr(d, "energy")])
    forces_ref = np.stack([d.force.numpy() for d in data_list])
    dipole_ref = np.stack([(d.dipole.numpy() if hasattr(d, "dipole") else np.zeros((1, 3))) for d in data_list]).squeeze()

    # --- Run model ---
    calc = MLAseCalculator(
        model_path=model_path,
        properties=["energy", "forces", "dipole"],
        device=device,
        precision="single"
    )

    energies_pred, forces_pred, dipole_pred = [], [], []
    for d in data_list:
        # Create ASE-like Atoms
        from ase import Atoms
        from ase.data import chemical_symbols
        atoms = Atoms(
            [chemical_symbols[z.item()] for z in d.z],
            positions=d.pos.numpy()
        )
        atoms.calc = calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        d_pred = atoms.calc.results.get("dipole", np.zeros(3))
        energies_pred.append(e)
        forces_pred.append(f)
        dipole_pred.append(d_pred)

    energies_pred = np.array(energies_pred)
    forces_pred = np.array(forces_pred)
    dipole_pred = np.array(dipole_pred)

    # --- Save data ---
    np.savez(
        os.path.join(output_dir, "predictions.npz"),
        energies_ref=energies_ref,
        energies_pred=energies_pred,
        forces_ref=forces_ref,
        forces_pred=forces_pred,
        dipole_ref=dipole_ref,
        dipole_pred=dipole_pred,
    )
    print(f"💾 Saved prediction results to {output_dir}/predictions.npz")

    # ===== Plotting =====
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.5, right=0.92, left=0.08, top=0.9, bottom=0.15)

    # Energy (A)
    hexbin_on_ax(
        axes[0], energies_ref, energies_pred,
        "Reference Energy (kcal/mol)", "Predicted Energy (kcal/mol)",
        "Energy Prediction", "A", scale=1.0, manual_axis=True
    )

    # Force (B)
    hexbin_on_ax(
        axes[1], forces_ref.ravel(), forces_pred.ravel(),
        r"Reference Force ($kcal/mol \cdot \AA$)", r"Predicted Force ($kcal/mol \cdot \AA$)",
        "Force Prediction", "B", scale=1.0, manual_axis=False
    )
    hexbin_on_ax(
        axes[2], dipole_ref.ravel(), dipole_pred.ravel(),
        r"Reference Dipole ($au$)", r"Predicted Dipole ($au$)",
        "Dipole Prediction", "C", scale=1.0, manual_axis=False
    )
    axes[2].set_aspect('equal', adjustable='box')
    out_file = os.path.join(output_dir, "energy_force_hexbin.png")
    fig.savefig(out_file, dpi=300)
    # plt.show()
    plt.close(fig)
    print(f"Saved plot to {out_file}")


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    model_path = "/scratch/moriya2524/mlp/tensornet_phbdi_tddft_transition_dipolemom_1000/training_15/models/best_model.pt"
    test_xyz = "/scratch/moriya2524/mlp/phbdi_tddft_datas/test/raw/test.xyz"
    output_dir = "/scratch/moriya2524/mlp/tensornet_phbdi_tddft_transition_dipolemom_1000/eval_results/"
    evaluate(model_path, test_xyz, output_dir, device="cuda")
