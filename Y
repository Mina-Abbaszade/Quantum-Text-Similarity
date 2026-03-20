import pandas as pd
import pickle

from lambeq.experimental.discocirc import DisCoCircReader
from lambeq import AtomicType, UnifyCodomainRewriter
from lambeq.backend.grammar import Ty, Box, Id
from lambeq.ansatz.circuit import Sim4Ansatz

# Types
N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE

n = Ty('n')
s = Ty('s')

reader = DisCoCircReader()

# Ansatz
ansatz = Sim4Ansatz(
    {N: 1, S: 2, P: 1},
    n_layers=3,
    n_single_qubit_params=3,
    discard=True
)

# -----------------------
# MERGE STRATEGIES
# -----------------------

def merge_global(diagram):
    """Single big merge box"""
    merger = UnifyCodomainRewriter(s)
    return merger(diagram)


def merge_stairs(diagram):
    """Iterative pairwise merging"""
    while len(diagram.cod) > 1:
        box = Box(name="merged_wire", dom=diagram.cod[:2], cod=s)
        diagram >>= (box @ Id(diagram.cod[2:]))
    return diagram


# -----------------------
# TEXT → CIRCUIT
# -----------------------

def text_to_diagram(text):
    return reader.text2circuit(
        text,
        rewrite_rules=(
            'determiner',
            'auxiliary',
            'noun_modification',
            'verb_modification'
        ),
        sandwich=True
    )


def process_text(text, merge_type="global"):
    diagram = text_to_diagram(text)

    if merge_type == "global":
        diagram = merge_global(diagram)
    elif merge_type == "stairs":
        diagram = merge_stairs(diagram)
    else:
        raise ValueError("merge_type must be 'global' or 'stairs'")

    return diagram


def diagram_to_circuit(diagram):
    return ansatz(diagram)


# -----------------------
# DATASET → CIRCUITS
# -----------------------

def generate_circuits(df, merge_type="global"):
    circuits_with_ids = []

    for _, row in df.iterrows():
        id_ = row["id"]
        text = row["synopsis"]

        diagram = process_text(text, merge_type)
        circuit = diagram_to_circuit(diagram)

        circuits_with_ids.append((id_, circuit))

    return circuits_with_ids


# -----------------------
# SAVE
# -----------------------

def save_circuits(circuits, path):
    with open(path, "wb") as f:
        pickle.dump(circuits, f)
