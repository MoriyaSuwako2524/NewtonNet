import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from newtonnet.utils.ase_interface import MLAseCalculator

# ===========================
# Helper for hexbin plotting
# ===========================
def hexbin_on_ax(ax, x, y, xlabel, ylabel, title, tag, scale=1.0, manual_axis=False):
    hb = ax.hexbin(x, y, gridsize=120, cmap='viridis', bins='log')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"({tag}) {title}", fontsize=13)
    ax.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', lw=1)
    ax.set_aspect('equal', adjustable='box')
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("log10(N)", fontsize=11)
    if manual_axis:
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([x.min(), x.max()])


# ===========================
# Evaluation function
# ===========================
def evaluate(model_path, test_xyz, output_dir, device="cuda", batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading model from {model_path}")
    print(f"Reading test structures from {test_xyz}")

    # ========== Load reference data ==========
    atoms_list = read(test_xyz, index=":")
    n_structs = len(atoms_list)
    print(f"Loaded {n_structs} test structures")

    energies_ref, forces_ref, dipole_ref = [], [], []
    for atoms in atoms_list:
        info = atoms.info
        energy = float(str(info.get("energy", 0)).replace(",", " "))
        energies_ref.append(energy)
        forces_ref.append(atoms.get_forces())
        dipole = None
        if "dipole" in info:
            dipole = np.array([float(x) for x in str(info["dipole"]).replace(",", " ").split()])
        else:
            # 尝试从 xyz header 解析
            dipole = np.zeros(3)
        dipole_ref.append(dipole)
    energies_ref = np.array(energies_ref)
    forces_ref = np.array(forces_ref)
    dipole_ref = np.array(dipole_ref)

    # ========== Run predictions ==========
    calc = MLAseCalculator(
        model_path=model_path,
        properties=["energy", "forces", "dipole"],
        precision="single",
        device=device,
        batch_size=batch_size,
    )

    energies_pred, forces_pred, dipole_pred = [], [], []
    for atoms in atoms_list:
        atoms.calc = calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        d = atoms.calc.results.get("dipole", np.zeros(3))
        energies_pred.append(e)
        forces_pred.append(f)
        dipole_pred.append(d)

    energies_pred = np.array(energies_pred)
    forces_pred = np.array(forces_pred)
    dipole_pred = np.array(dipole_pred)

    # ========== Save numerical comparison ==========
    np.savez(
        os.path.join(output_dir, "predictions.npz"),
        energies_ref=energies_ref,
        energies_pred=energies_pred,
        forces_ref=forces_ref,
        forces_pred=forces_pred,
        dipole_ref=dipole_ref,
        dipole_pred=dipole_pred,
    )
    print(f"Saved prediction results to {output_dir}/predictions.npz")

    # ========== Plot comparison ==========
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.4, left=0.07, right=0.93, bottom=0.15, top=0.9)

    # Energy (A)
    hexbin_on_ax(
        axes[0],
        energies_ref,
        energies_pred,
        "Reference Energy (kcal/mol)",
        "Predicted Energy (kcal/mol)",
        "Energy Prediction",
        "A",
        manual_axis=True,
    )

    # Force (B)
    hexbin_on_ax(
        axes[1],
        forces_ref.ravel(),
        forces_pred.ravel(),
        r"Reference Force ($kcal/mol \cdot \AA$)",
        r"Predicted Force ($kcal/mol \cdot \AA$)",
        "Force Prediction",
        "B",
    )

    # Dipole (C)
    hexbin_on_ax(
        axes[2],
        dipole_ref.ravel(),
        dipole_pred.ravel(),
        r"Reference Dipole ($e \cdot bohr$)",
        r"Predicted Dipole ($e \cdot bohr$)",
        "Dipole Prediction",
        "C",
    )
    axes[2].set_aspect("equal", adjustable="box")

    out_file = os.path.join(output_dir, "energy_force_dipole_hexbin.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"✅ Saved plot to {out_file}")


# ===========================
# Main execution
# ===========================
if __name__ == "__main__":
    model_path = "/scratch/moriya2524/mlp/tensornet_phbdi_tddft_transition_dipolemom_1000/training_5/models/best_model.pt"
    test_xyz = "/scratch/moriya2524/mlp/phbdi_tddft_datas/test/raw/test.xyz"
    output_dir = "/scratch/moriya2524/mlp/tensornet_phbdi_tddft_transition_dipolemom_1000/eval_results/"
    evaluate(model_path, test_xyz, output_dir, device="cuda", batch_size=8)
