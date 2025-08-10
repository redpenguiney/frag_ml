import os
import pandas
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem import Recap, Descriptors, Lipinski, Crippen
from rdkit.Chem.Fraggle import FraggleSim
import pathlib
import multiprocessing
import ctypes
import numpy
import argparse
from unidock_tools.modules.docking.unidock import run_unidock 

# Used to pass various command-line arguments to the worker processes.
class Config:
    def __init__(self):
        self.BOX_SIZES = None
        self.CENTER_COORDS = None
        self.PROTEIN_PDBQT_PATH = None

def smiles_to_mol(smiles):

    smiles = smiles.replace('*', 'C')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    AllChem.ComputeGasteigerCharges(mol, nIter=12, throwOnParamFailure=False)
    return mol

def validate_molecule(mol):
    return mol is not None and mol.GetNumAtoms() > 2 and mol.GetNumBonds() > 0

def is_multi_molecule(mol):
    return len(Chem.GetMolFrags(mol)) > 1

def is_trivial_fragment(mol):
    heavy = mol.GetNumHeavyAtoms() # number of non-hydrogens
    if heavy < 3:
        return True
    if all(a.GetAtomicNum() == 6 for a in mol.GetAtoms()) and heavy <= 4:
        return True
    return False

def sanitize_dummy_atoms(mol):
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom like [*]
            atom.SetAtomicNum(6)      # Replace with C
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return mol    

def mol_to_smiles(mol, finalize=False):
    if finalize:
        mol = sanitize_dummy_atoms(mol)
        mol = Chem.RemoveHs(mol)
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
    return Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=not finalize, canonical=finalize)

class fragment:
    def __init__(self, molecule, algorithmSource):
        self.src = algorithmSource
        self.mol = molecule
        self.file = "???"
        assert(self.mol)

    def __repr__(self):
        return mol_to_smiles(self.mol, True) + " from " + self.file
        

def get_recap_frags(molecule):
    recap_tree = Recap.RecapDecompose(molecule)
    fragments = []

    if recap_tree:
        leaves = recap_tree.GetLeaves()
        if leaves:
            for smile, node in leaves.items():
                # Properly handle wildcard atoms
                cleaned_smile = smile.replace('*', 'C')  # Replace wildcard with carbon
                fragments.append(fragment(smiles_to_mol(cleaned_smile), "recap"))
        else:
            print("No leaves found in the Recap tree.")
    else:
        print("Failed to obtain Recap decomposition.")
    
    return fragments

# Used to supplies global variables to each process.
def init_child_process(fResults, nComplete, totalToComplete, lastSdf, config):
    global finalResults
    finalResults = fResults
    global numComplete
    numComplete = nComplete
    global total
    total = totalToComplete

    global LAST_SDF_NAME
    LAST_SDF_NAME = lastSdf

    global CONFIG
    CONFIG = config

# Processes a chunk of input data.
def process_csv_chunk(csv_data, ):
    results = []

    global LAST_SDF_NAME
    global total
    global numComplete
    global finalResults
    global CONFIG

    for index, row in csv_data.iterrows():
     
        drug_smiles = row["drug"]

        try:
            drug_molecule = smiles_to_mol(drug_smiles)

            # First, we generate a bunch of fragments of the drug molecule. 
            potential_fragments = []

            brics_result = BRICS.BRICSDecompose(drug_molecule, returnMols=True, singlePass=True)
            converted_brics_result = []
            for frag in brics_result:
                converted_brics_result.append(fragment(frag, "brics"))
            for frag in converted_brics_result:
                potential_fragments.append(frag)

            fraggle_result = FraggleSim.generate_fraggle_fragmentation(drug_molecule)
            for i in range(len(fraggle_result)):
                fraggle_result[i] = fragment(smiles_to_mol(fraggle_result[i]), "fraggle")
                potential_fragments.append(fraggle_result[i])

            recap_result = get_recap_frags(drug_molecule)
            potential_fragments.extend(recap_result)

            # eliminate fragments based on rule of 3
            i = 0
            while i < len(potential_fragments):
                frag = potential_fragments[i].mol

                if not validate_molecule(frag) or is_multi_molecule(frag) or is_trivial_fragment(frag):
                    del potential_fragments[i]
                    continue

                weight = Descriptors.MolWt(frag)
                num_hbond_acceptors = Lipinski.NumHAcceptors(frag)
                num_hbond_donators = Lipinski.NumHDonors(frag)
                if weight >= 300 or num_hbond_acceptors > 3 or num_hbond_donators > 3: 
                    del potential_fragments[i]
                    continue
                
                clogp = Crippen.MolLogP(frag)
                if clogp >= 3:
                    del potential_fragments[i]
                    continue

                i += 1

            
            ligand_paths = []

            # Next, we dock every fragment.
            # We write .sdf files for each fragment because unidock only accepts .sdf files as ligand inputs.

            prev_start = -1

            with LAST_SDF_NAME.get_lock():
                for i in range(LAST_SDF_NAME.value + 1, LAST_SDF_NAME.value + 1 + len(potential_fragments)):
                    path = os.getcwd()+'/sdf_input/f'+str(i)+'.sdf'
                    w = Chem.SDWriter(os.getcwd()+'/sdf_input/f'+str(i)+'.sdf')
                    w.write(potential_fragments[i-1-LAST_SDF_NAME.value].mol)
                    w.close()
                    potential_fragments[i-1-LAST_SDF_NAME.value].file = path
                    ligand_paths.append(pathlib.Path(path))

                prev_start = LAST_SDF_NAME.value
                LAST_SDF_NAME.value += len(potential_fragments)

            # perform docking
            _, scoreArrays = run_unidock(
                receptor = CONFIG.PROTEIN_PDBQT_PATH, 
                ligands = ligand_paths,
                output_dir = pathlib.Path(os.getcwd()+'/docking_output/'), 
                center_x = CONFIG.CENTER_COORDS[0], 
                center_y = CONFIG.CENTER_COORDS[1], 
                center_z = CONFIG.CENTER_COORDS[2], 
                size_x = CONFIG.BOX_SIZES[0], 
                size_y = CONFIG.BOX_SIZES[1], 
                size_z = CONFIG.BOX_SIZES[2],
                #score_only = True, for some reason this flag caps sizes to something stupidly small?
                num_modes = 1,
                exhaustiveness = 384,
                max_step = 40,
            )

            scores = []
        
            assert(len(scoreArrays) == len(ligand_paths))
            for arr in scoreArrays:
                assert(len(arr) == 1)
                scores.extend(arr)

            # We select the best fragment based on docking score.
            best_fragment = None
            best_score = 999999
            i = 0
            for frag in potential_fragments:
                docking_score = scores[i]
                i+=1

                if docking_score < best_score:
                    
                    best_fragment = frag
                    best_score = docking_score
                

            drug_smiles = mol_to_smiles(drug_molecule, True)
            if best_fragment:
                results.append({'drug' : drug_smiles, 'fragment' : mol_to_smiles(best_fragment.mol, True), 'fragment_src' : best_fragment.src, 'fragment_mass' : Descriptors.MolWt(best_fragment.mol), "docking_score" : best_score })
            else:
                print("Warning: failure to find suitable fragment for drug ",  drug_smiles, " at index ", index)

            with numComplete.get_lock():
                numComplete.value+=1
                print("Progress ", numComplete.value , "/", total)

        except Exception as e:
            with numComplete.get_lock():
                print("Failure at index ", index, ": ", e)
                numComplete.value+=1
                print("Progress ", numComplete.value , "/", total)

    finalResults.extend(results)



def main(src_file, out_file, config):

    # Multiprocessing is used to maximize GPU usage. 
    # NOTE: you might want to increase this number if you are not getting full GPU utilization.
    num_processes = 4

    mpManager = multiprocessing.Manager()

    chunk_size = 1000 * num_processes


    finalResults = mpManager.list([])

    NUM_ROWS = 10000000

    csv_data = pandas.read_csv(src_file, chunksize=chunk_size, nrows=NUM_ROWS)

    numComplete = multiprocessing.Value(ctypes.c_int32, 0)
    total = len(pandas.read_csv(src_file, nrows=NUM_ROWS))

    LAST_SDF_NAME = multiprocessing.Value(ctypes.c_int32, -1)

    # Populate finalResults with best fragment/drug data (as well as some other data about the nature of the fragments selected).
    pool = multiprocessing.Pool(num_processes, initializer=init_child_process, initargs=(finalResults, numComplete, total, LAST_SDF_NAME, config))
    count = 0
    for file_chunk in csv_data:
        line = count * chunk_size
        print(f"Processing {chunk_size} lines after line {line}")
        
        pool.map(process_csv_chunk, numpy.array_split(file_chunk, num_processes))

        count += 1

    pool.close()
    pool.join()

    # Write output to file
    results_df = pandas.DataFrame(list(finalResults))
    results_df.to_csv(out_file, index=False)

    with numComplete.get_lock():
        assert(numComplete.value == total)
        print("Generated data for ", len(finalResults), " fragment-drug pairs out of ", total)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Drug fragmentation, cleanup, and docking for ML dataset generation')

    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file with SMILES strings')
    parser.add_argument('--pdbqt_path', type=str, required=True, help='Path to mol2 file')
    parser.add_argument('--center_coords', type=float, required=True, nargs=3, help='Center coordinates for docking box (X Y Z)')
    parser.add_argument('--box_sizes', type=float, required=True, nargs=3, help='Box sizes for docking (X Y Z) in angstroms')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the results CSV')

    args = parser.parse_args()

    config = Config()
    config.BOX_SIZES = args.box_sizes
    config.CENTER_COORDS = args.center_coords
    config.PROTEIN_PDBQT_PATH = args.pdbqt_path
    main(args.input_csv, args.output_path, config)

