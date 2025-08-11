import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Recap, Descriptors, Lipinski, Crippen, QED, rdMolDescriptors
from rdkit.Chem.Fraggle import FraggleSim
from rdkit.Chem.GraphDescriptors import BertzCT
from tqdm import tqdm
from rdkit.Contrib.SA_Score import sascorer
import argparse

# === Helper Functions ===
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles.replace('*', 'C'))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    AllChem.ComputeGasteigerCharges(mol, nIter=12, throwOnParamFailure=False)
    return mol

def sanitize_dummy_atoms(mol):
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return mol

def clean_smiles(mol):
    mol = sanitize_dummy_atoms(mol)
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)

def get_recap_frags(molecule):
    recap_tree = Recap.RecapDecompose(molecule)
    fragments = []
    if recap_tree:
        leaves = recap_tree.GetLeaves()
        for smile, node in leaves.items():
            mol = Chem.MolFromSmiles(smile.replace('*', 'C'))
            if mol:
                fragments.append(Fragment(mol, "recap"))
    return fragments

def validate_molecule(mol):
    return mol is not None and mol.GetNumAtoms() > 2 and mol.GetNumBonds() > 0

def is_multi_molecule(mol):
    return len(Chem.GetMolFrags(mol)) > 1

def is_trivial_fragment(mol):
    heavy = mol.GetNumHeavyAtoms()
    if heavy < 3:
        return True
    if all(a.GetAtomicNum() == 6 for a in mol.GetAtoms()) and heavy <= 4:
        return True
    return False

def compute_fragment_score(mol):
    try:
        qed = QED.qed(mol)
        mw = Descriptors.MolWt(mol)
        sas = sascorer.calculateScore(mol)
        complexity = BertzCT(mol)
        pharm_feats = len(rdMolDescriptors.GetMorganFingerprint(mol, 2).GetNonzeroElements())
        score = ( # see methods in paper for explanation of scoring function
            3.0 * qed
            - 0.5 * mw
            - 2.0 * sas
            + 1.5 * complexity
            + 2.0 * pharm_feats
        )
        return score
    except Exception as e:
        # print(f"[Score ERROR] {e}")
        return -float('inf')

class Fragment:
    def __init__(self, molecule, algorithmSource):
        self.src = algorithmSource
        self.mol = molecule

# === Command Line Arguments ===

parser = argparse.ArgumentParser(description='Drug fragmentation, cleanup, and docking script')
parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file with SMILES strings')
parser.add_argument('--output_path', type=str, required=True, help='Output path for the results CSV')
args = parser.parse_args()
input_path = args.input_csv
output_path = args.output_path

# === Main Loop ===

csv_data = pd.read_csv(input_path, header=None)
results = []

for index, row in tqdm(csv_data.iterrows(), total=len(csv_data)):
    try:
        drug_smiles = row[0]
        raw_mol = Chem.MolFromSmiles(drug_smiles)
        if raw_mol is None:
            continue

        drug_molecule = smiles_to_mol(drug_smiles)
        potential_fragments = []

        frag_by_method = {"brics": [], "fraggle": [], "recap": []}

        # BRICS
        brics_result = BRICS.BRICSDecompose(drug_molecule, returnMols=True, singlePass=True)
        for frag in brics_result:
            if validate_molecule(frag) and not is_trivial_fragment(frag) and not is_multi_molecule(frag):
                f = Fragment(frag, "brics")
                potential_fragments.append(f)
                frag_by_method["brics"].append(clean_smiles(frag))

        # Fraggle
        try:
            fraggle_result = FraggleSim.generate_fraggle_fragmentation(drug_molecule)
            if isinstance(fraggle_result, list):
                for frag in fraggle_result:
                    frag = smiles_to_mol(frag)
                    if validate_molecule(frag) and not is_trivial_fragment(frag) and not is_multi_molecule(frag):
                        f = Fragment(frag, "fraggle")
                        potential_fragments.append(f)
                        frag_by_method["fraggle"].append(clean_smiles(frag))
        except Exception as e:
            print(f"[Fraggle ERROR]: {e}")

        # Recap
        recap_frags = get_recap_frags(drug_molecule)
        for frag in recap_frags:
            if validate_molecule(frag.mol) and not is_trivial_fragment(frag.mol) and not is_multi_molecule(frag.mol):
                potential_fragments.append(frag)
                frag_by_method["recap"].append(clean_smiles(frag.mol))

        best_fragment = None
        best_score = -float('inf')
        fallback_fragment = None
        fallback_qed = -float('inf')

        for frag in potential_fragments:
            sm = clean_smiles(frag.mol)
            score = compute_fragment_score(frag.mol)
            if score > best_score:
                best_score = score
                best_fragment = frag
            try:
                qed = QED.qed(frag.mol)
                if qed > fallback_qed:
                    fallback_qed = qed
                    fallback_fragment = frag
            except:
                continue

        if best_fragment is None and fallback_fragment:
            best_fragment = fallback_fragment
            best_score = fallback_qed
            print(f"All scoring failed, falling back to QED-only fragment: {clean_smiles(best_fragment.mol)}")

        if best_fragment:
            results.append({
                'drug': drug_smiles,
                'fragment': clean_smiles(best_fragment.mol),
                'fragment_src': best_fragment.src,
                'score': best_score
            })

    except Exception as e:
        print(f"[Error @ index {index}]: {e}")
        continue

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"\nSaved {len(results)} best fragments.")