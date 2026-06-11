# 🧬 Predykcja Bioaktywności EGFR

Projekt z przedmiotu **Warsztaty AI** — system uczenia maszynowego do przewidywania aktywności leków względem białka EGFR.

---

## O projekcie

EGFR (*Epidermal Growth Factor Receptor*) to kluczowy cel terapeutyczny w leczeniu nowotworów. Testowanie milionów związków chemicznych eksperymentalnie jest kosztowne i czasochłonne — modele ML pozwalają przewidzieć aktywność cząsteczki zanim trafi do laboratorium.

W projekcie porównuję dwa podejścia:

| Model | Wejście | Test R² |
|-------|---------|---------|
| **MLP** | Odciski Morgana (2048-bit) | 0.55 |
| **GNN — GIN** ✅ | Graf cząsteczkowy (23-dim) | **0.60** |

Oba modele trenowane na **16 441 pomiarach bioaktywności** z bazy [ChEMBL 36](https://www.ebi.ac.uk/chembl/).

---

## Architektura

### MLP
```
SMILES → Odcisk Morgana (2048-bit) → Dense(512) → Dense(128) → pIC50
```
Dropout 0.3 · Adam lr=0.001 · early stopping

### GNN — GIN (Graph Isomorphism Network)
```
SMILES → Graf cząsteczkowy → 4× GINConv(256) + BatchNorm → Global Mean Pool → FC → pIC50
```
GIN używa agregacji sumy zamiast uśredniania (bardziej ekspresywny niż GCN) · Dropout 0.2 · gradient clipping

---

## Wyniki

```
MLP  —  Test R²: 0.5507  |  RMSE: 0.8453
GNN  —  Test R²: 0.5956  |  RMSE: 0.8404  ✅
```

- Wymaganie: R² > 0.5 ✅
- Brak przeuczenia — Val R² ≈ Test R²
- GNN bije przykład prowadzącego (0.547) o +8.8%

---

## Interfejs

Aplikacja Streamlit do przewidywania aktywności na żywo:

```bash
source venv_312/bin/activate
streamlit run app.py
```

Wpisz SMILES, kliknij **Predict** — otrzymasz predykcje MLP i GNN, strukturę 2D oraz właściwości cząsteczki (MW, LogP, HBA, HBD).

Przykład: `CC(C)Cc1ccc(cc1)C(C)C(O)=O` (ibuprofen)

---

## Dane

- **Źródło:** ChEMBL 36, cel EGFR (CHEMBL203)
- **Surowe:** 21 626 pomiarów IC50 → **po czyszczeniu:** 16 441 związków
- **Podział:** 80% trening / 10% walidacja / 10% test
- **Metryka:** pIC50 = −log₁₀(IC50) — wyższa wartość = silniejszy związek

---

## Struktura projektu

```
├── app.py                              # Interfejs Streamlit
├── train_gnn_improved.py               # Skrypt trenowania GNN
├── scripts/
│   ├── model.py                        # Architektura MLP
│   ├── gnn_model_improved.py           # Architektura GIN
│   ├── featurize.py                    # Wektoryzacja (RDKit)
│   └── utils.py                        # Odciski Morgana, podział danych
└── eda_chembl36_kowalczyk_kasia.ipynb  # Analiza eksploracyjna danych
```

---

*Kasia Kowalczyk · Warsztaty AI · 2026*
