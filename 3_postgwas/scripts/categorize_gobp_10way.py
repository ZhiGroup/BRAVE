from pathlib import Path

import pandas as pd

_GENESET = Path(__file__).resolve().parents[1] / "results" / "geneset"
IN  = str(_GENESET / 'gobp_cross_region_top.csv')
OUT = str(_GENESET / 'gobp_terms_categorized.csv')

df = pd.read_csv(IN)

CATEGORIES = [
    ("Structural/ECM/Glial", [
        "mesenchyme", "extracellular matrix", "collagen", "fibril", "proteoglycan",
        "basement membrane", "ecm", "glial", "astrocyte", "oligodendrocyte", "myelination",
        "schwann", "ependymal", "fibroblast", "leptomeningeal", "myelin",
        "cartilage", "bone morphogen", "ossification", "connective tissue",
        "endothelial", "vascular", "angiogenesis", "blood vessel",
        "microglia", "macrophage",
    ]),
    ("Neuronal/Axon/Synapse", [
        "neuron", "neuronal", "axon", "dendrite", "synapse", "synaptic",
        "neurotransmitter", "action potential", "long-term potentiation",
        "nervous system", "neural", "neuropil", "neurogenesis",
        "gluta", "gaba", "dopamine", "serotonin", "acetylcholine",
        "neurite", "postsynaptic", "presynaptic",
    ]),
    ("Brain Region Dev.", [
        "forebrain", "telencephalon", "pallium", "subpallium",
        "dentate gyrus", "hippocampus", "cerebellum", "cerebellar",
        "olfactory lobe", "pituitary", "diencephalon", "hypothalamus",
        "striatum", "amygdala", "thalamus",
        "neocortex", "cortex development", "corticogenesis",
    ]),
    ("Morphogenesis/Dev.", [
        "morphogenesis", "organogenesis", "pattern specification",
        "embryo", "embryonic", "regionalization", "body plan",
        "somitogenesis", "gastrulation", "segmentation", "induction",
        "anterior posterior", "dorsal ventral", "left right",
        "cell fate", "cell differentiation", "differentiation",
        "determination", "specification", "commitment",
        "developmental growth", "multicellular organism growth",
    ]),
    ("WNT/BMP/Notch Signaling", [
        "wnt", "bmp", "notch", "hedgehog", "tgf", "smad",
        "bone morphogenetic", "canonical wnt", "non canonical wnt",
        "planar polarity", "frizzled",
    ]),
    ("MAPK/RAS/GTPase Signaling", [
        "mapk", "ras ", "rho ", "cdc42", "gtpase", "erk",
        "map kinase", "ras protein", "rho protein", "small gtpase",
        "semaphorin", "roundabout", "plexin",
        "receptor tyrosine kinase", "tyrosine kinase signaling",
        "enzyme linked receptor", "transmembrane receptor protein tyrosine",
        "phosphorylation", "kinase", "phosphorus metabolic",
    ]),
    ("Cell Cycle/Proliferation", [
        "cell cycle", "mitotic", "cell division", "proliferation",
        "cell growth", "growth regulation", "regulation of growth",
        "cellular senescence", "apoptot", "cell death",
        "cell size", "cell population",
    ]),
    ("Transcription/Chromatin", [
        "chromatin", "transcription", "rna biosynthetic", "rna metabolic",
        "gene expression", "dna binding", "epigenetic",
        "histone", "post transcriptional", "mrna processing",
        "nucleobase", "pyrimidine", "purine",
    ]),
    ("Cytoskeleton/Adhesion/Migration", [
        "actin", "cytoskeleton", "focal adhesion", "lamellipodium",
        "supramolecular fiber", "cell junction", "cell matrix adhesion",
        "cell substrate junction", "cell motility", "locomotion",
        "taxis", "chemotaxis", "tissue migration", "cell polarity",
        "polarity", "microtubule",
    ]),
    ("Cardiovascular/Immune/Other Organ", [
        "heart", "cardiac", "aorta", "circulatory", "cardiovascular",
        "kidney", "renal", "nephron", "metanephros",
        "hemopoiesis", "monocyte", "granulocyte", "immune",
        "pancreas", "mammary", "gland development",
        "smooth muscle tissue", "endoderm",
    ]),
]

def classify(label):
    lbl = label.lower()
    for cat, kws in CATEGORIES:
        if any(kw in lbl for kw in kws):
            return cat
    return "Other/Metabolism"

df['category_refined'] = df['label'].apply(classify)

# Rename original 4-way column and select output columns
df = df.rename(columns={'category': 'category_4way'})
cols = ['VARIABLE', 'label', 'category_4way', 'category_refined',
        'n_sig', 'n_regions', 'mean_log10p', 'max_log10p', 'min_fdr_q']
df[cols].to_csv(OUT, index=False)

print(df['category_refined'].value_counts().to_string())
print(f"\nTotal: {len(df)} terms")
print(f"Saved: {OUT}")
