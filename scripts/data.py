import numpy as np
import os

# === Define atomic number → symbol map ===
atom_charge_dict = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,
    "Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,
    "Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,
    "Se":34,"Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,"Tc":43,
    "Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,"Sb":51,"Te":52,"I":53,"Xe":54
}
# invert mapping:  {atomic_number: symbol}
Z2symbol = {v: k for k, v in atom_charge_dict.items()}

# ==== Input files ====
path = "/scratch/moriya2524/mlp/phbdi_tddft_datas/"
coord = np.load(os.path.join(path, "full_coord.npy"))   # (nframes, natoms, 3)
force = np.load(os.path.join(path, "full_force.npy"))   # (nframes, natoms, 3)
energy = np.load(os.path.join(path, "full_S1_energy.npy"))  # (nframes,)
transmom = np.load(os.path.join(path, "full_transmom.npy"))  # (nframes, 3)
types = np.load(os.path.join(path, "full_type.npy"))    # (natoms,)
split = np.load(os.path.join(path, "1000_split.npz"))   # contains idx_train, idx_val, idx_test

# convert to symbols
symbols = [Z2symbol[int(Z)] for Z in types]

# === Helper to write ASE-style xyz ===
def write_xyz(filename, indices):
    with open(filename, "w") as f:
        for i in indices:
            natoms = len(symbols)
            tm = transmom[i]
            f.write(f"{natoms}\n")
            f.write(
                f'Properties=species:S:1:pos:R:3:forces:R:3 energy={energy[i]} '
                f'dipole={tm[0]:.8f},{tm[1]:.8f},{tm[2]:.8f} pbc="F F F"\n'
            )
            for a in range(natoms):
                atom = symbols[a]
                x, y, z = coord[i, a]
                fx, fy, fz = force[i, a]
                f.write(f"{atom:2s} {x:15.8f} {y:15.8f} {z:15.8f} {fx:15.8f} {fy:15.8f} {fz:15.8f}\n")

# === Write train/val/test sets ===
write_xyz(os.path.join(path, "train.xyz"), split["idx_train"])
write_xyz(os.path.join(path, "val.xyz"), split["idx_val"])
write_xyz(os.path.join(path, "test.xyz"), split["idx_test"])

print("✅ Done! Written: train.xyz, val.xyz, test.xyz")
