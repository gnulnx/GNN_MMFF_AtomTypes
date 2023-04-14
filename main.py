import click

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import re


def mol_from_qm9_data(data):
    mol = Chem.Mol()
    edit_mol = Chem.EditableMol(mol)
    for row in data.x:
        atom = Chem.Atom(int(row[5]))
        edit_mol.AddAtom(atom)

    bonds = set()
    for bdx, edge in enumerate(data.edge_index.T):
        
        edge = sorted(edge)
        
        if str(edge) in bonds:
            continue

        if data.edge_attr[bdx][0]:
            edit_mol.AddBond(int(edge[0]), int(edge[1]), Chem.rdchem.BondType.SINGLE)
        elif data.edge_attr[bdx][1]:
            edit_mol.AddBond(int(edge[0]), int(edge[1]), Chem.rdchem.BondType.DOUBLE)
        elif data.edge_attr[bdx][2]:
            edit_mol.AddBond(int(edge[0]), int(edge[1]), Chem.rdchem.BondType.TRIPLE)
        elif data.edge_attr[bdx][3]:
            edit_mol.AddBond(int(edge[0]), int(edge[1]), Chem.rdchem.BondType.AROMATIC)
        
        bonds.add(str(edge))

    mol = edit_mol.GetMol()
    # Must call these two before working with forcefields
    try:
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        mol = Chem.AddHs(mol)
    except rdkit.Chem.rdchem.AtomValenceException:
        print("SKipping........")
        return None

    return mol

def CalcMMFF94AtomTypes():
    dset = QM9(".")

    if os.path.exists("mmff94_mol.txt"):
        os.remove("mmff94_mol.txt")

    with open("mmff94_mol.txt", 'a') as f:
        f.write("# Basic format of the file.\n")
        f.write("# Mol:  qm9-name=name smiles=smiles\n")
        f.write("# atomic_number, mmff94_atom_type, mmff94_partial_charge\n")
        f.write("# atomic_number, mmff94_atom_type, mmff94_partial_charge\n")
        f.write("# ......\n")
        for idx, data in enumerate(dset):
            print(idx, "of", len(dset))

            mol = mol_from_qm9_data(data)
            if not mol:
                continue

            smiles = Chem.MolToSmiles(mol)

            output = [f"Mol: qm9-name={data.name} smiles={smiles}\n"]

            props = AllChem.MMFFGetMoleculeProperties(mol)
            for idx, a in enumerate(mol.GetAtoms()):
                output.append(f"a, {a.GetAtomicNum()}, {props.GetMMFFAtomType(idx)}, {props.GetMMFFPartialCharge(idx)}" + "\n")
            
            f.writelines(output)

            f.writelines("edge_index\n")
            for row in data.edge_index.T.tolist():
                f.writelines(f"ei, {str(row[0])}, {str(row[1])}" + "\n")

            f.writelines("edget_attr\n")
            for row in data.edge_attr:
                f.writelines(f"er, {int(row[0].item())}, {int(row[1].item())}, {int(row[2].item())}, {int(row[3].item())}" + "\n")
    
def read_mm94_type_file(filename="MMFF94Types.txt"):

    results = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            atm_env, desc = line.split(":")
            atm_env = atm_env.rstrip()

            match = re.search(r"\(MMFF number (\d+)\) (\w+.*)", desc)

            if match:
                results[int(match.group(1))] = f"{atm_env} - {match.group(2)}"
            else:
                print("no match")
                input()

    
    
    return results

def build_atom_type_stats(fielname="mmff94_mol.txt"):

    mmff94_types = read_mm94_type_file()
    for key, val in mmff94_types.items():
        print(key, val, type(key))
    input()

    fftypes = {}
    atom_numbers = {}
    with open(fielname, 'r') as f:
        for line in f:

            if line.startswith("a"):
                # print(line)
                _, atomic_number, fftype, change, = line.split(",")
                if fftype not in fftypes:
                    fftypes[fftype] = 0
                fftypes[fftype] += 1

                if atomic_number not in atom_numbers:
                    atom_numbers[atomic_number] = 0
                atom_numbers[atomic_number] += 1
    
    sorted_fftypes = {k: v for k, v in sorted(fftypes.items(), key=lambda item: item[1], reverse=True)}
    print("fftype, count, %, description")

    total = 0
    for key, val in sorted_fftypes.items():
        total += val

    for key, val in sorted_fftypes.items():
        key = int(key)
        print(f"{key}, {val}, {val/total*100:.3f}%, {mmff94_types[key]}")

@click.command()
@click.option('--build-atom-type-file', is_flag=True, help='Constrcut atom type file.')
@click.option('--show-atom-type-stats', is_flag=True, help='atom type stats.')
def main(build_atom_type_file, show_atom_type_stats):
    if build_atom_type_file:
        CalcMMFF94AtomTypes()

    if show_atom_type_stats:
        build_atom_type_stats()
        

if __name__ == '__main__':
    main()