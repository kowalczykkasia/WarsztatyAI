#!/usr/bin/env python3
"""Simple Streamlit UI for EGFR bioactivity prediction."""

import sys
from pathlib import Path
import streamlit as st
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen, Lipinski

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.model import IC50MLP
from scripts.gnn_model_improved import IC50GIN
from scripts.featurize import smiles_to_graph, NODE_FEATURE_DIM
from scripts.utils import smiles_to_morgan_fp

st.set_page_config(page_title="EGFR pIC50 Predictor", layout="wide")
st.title("🧬 EGFR Bioactivity Predictor")
st.markdown("Predict drug activity (pIC50) for EGFR target using MLP and GNN models")

# Load models
@st.cache_resource
def load_models():
    device = torch.device("cpu")

    # MLP
    mlp = IC50MLP(input_dim=2048, hidden_dims=[512, 128], dropout_rate=0.3)
    mlp.load_state_dict(torch.load("best_model.pt", map_location=device))
    mlp.eval()

    # GNN
    gin = IC50GIN(node_features=NODE_FEATURE_DIM, hidden_dim=256, dropout_rate=0.2)
    gin.load_state_dict(torch.load("best_gnn_model_random_improved.pt", map_location=device))
    gin.eval()

    return mlp, gin, device

# Prediction functions
def predict_mlp(smiles, model, device):
    try:
        fp = smiles_to_morgan_fp(smiles, radius=2, n_bits=2048)
        if fp is None:
            return None
        x = torch.tensor([fp], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(x).item()
        return pred
    except:
        return None

def predict_gnn(smiles, model, device):
    try:
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None
        graph = graph.to(device)
        with torch.no_grad():
            pred = model(graph).item()
        return pred
    except:
        return None

def get_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "MW": f"{Descriptors.MolWt(mol):.2f}",
            "LogP": f"{Crippen.MolLogP(mol):.2f}",
            "HBA": Lipinski.NumHAcceptors(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "RotBonds": Lipinski.NumRotatableBonds(mol),
        }
    except:
        return None

# Load models
try:
    mlp, gin, device = load_models()
    models_ok = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_ok = False

if models_ok:
    # Input
    col1, col2 = st.columns([4, 1])
    with col1:
        smiles = st.text_input("Enter SMILES:", placeholder="e.g., CC(C)Cc1ccc(cc1)C(C)C(O)=O")
    with col2:
        predict_btn = st.button("Predict", use_container_width=True)

    if predict_btn and smiles.strip():
        with st.spinner("Predicting..."):
            mlp_pred = predict_mlp(smiles, mlp, device)
            gnn_pred = predict_gnn(smiles, gin, device)
            props = get_properties(smiles)

            if mlp_pred or gnn_pred:
                # Results
                st.markdown("### 📊 Predictions")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if mlp_pred:
                        st.metric("MLP pIC50", f"{mlp_pred:.3f}")
                    else:
                        st.metric("MLP", "❌ Failed")

                with col2:
                    if gnn_pred:
                        st.metric("GNN pIC50", f"{gnn_pred:.3f}")
                    else:
                        st.metric("GNN", "❌ Failed")

                with col3:
                    if mlp_pred and gnn_pred:
                        avg = (mlp_pred + gnn_pred) / 2
                        st.metric("Average", f"{avg:.3f}")

                # Properties
                if props:
                    st.markdown("### 🧪 Molecular Properties")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("MW", props["MW"])
                    col2.metric("LogP", props["LogP"])
                    col3.metric("HBA", props["HBA"])
                    col4.metric("HBD", props["HBD"])
                    col5.metric("RotBonds", props["RotBonds"])

                # Structure
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        st.markdown("### 🔬 Molecular Structure")
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, width=300)
                except:
                    pass
            else:
                st.error("❌ Failed to predict - invalid SMILES or model error")

    # Info
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About
    - **MLP**: Multi-Layer Perceptron (Morgan fingerprints, R² = 0.55)
    - **GNN**: Graph Neural Network - GIN (R² = 0.60 on EGFR)
    - **Target**: EGFR bioactivity
    - **pIC50**: -log₁₀(IC50) - Higher = More potent
    """)
else:
    st.error("Could not load models")
