# COS 435 Final Project — Language Emergence in Social Learning Agents

A replication and extension of [*A framework for the emergence and analysis of language in social learning agents*](https://www.nature.com/articles/s41467-024-53277-3) (Guo et al., 2024). We reproduce the core experiments and add novel analyses using a self-contained pipeline.

## Setup

```bash
git clone https://github.com/keverage-guan/cos435_final_project
cd cos435_final_project
pip install -r requirements.txt
```

## Running the Pipeline

All experiments are run from the `language_emergence/` directory via `main.py`:

```bash
cd language_emergence
python main.py --stage <stage>
```

### Stages (run in order)

| Stage | What it does |
|---|---|
| `generate_data` | Generate teacher task definitions (wall configs + label dicts) |
| `generate_teacher` | Load/generate teacher Q-matrices (`--regen` to regenerate from scratch) |
| `train_language` | Train the sparse autoencoder (SAE) on teacher Q-matrices |
| `train_student` | Train baseline student agent (no feedback) |
| `feedback` | Joint feedback training — SAE + student trained together |
| `telephone_game` | Run telephone game across generations (gen2–5) |
| `figure3` | PCA analysis, topographic similarity, entropy (Fig. 3 + Table 5) |
| `figure4` | Generalization experiment — 7 goal groups × 5 languages (Fig. 4 + Table 6) |
| `all` | Run all stages sequentially |

### Fully self-contained run (no external data needed)

```bash
python main.py --stage all --regen
```

This generates all data from scratch and runs the full pipeline end-to-end. Intermediate checkpoints are saved to `outputs/checkpoints/` so individual stages can be re-run without repeating earlier steps.

### Outputs

| Location | Contents |
|---|---|
| `outputs/checkpoints/` | Model weights and intermediate data (gitignored) |
| `outputs/figures/` | All figures (`fig_telephone_game.png`, `fig3_*.png`, `fig4_*.png`) |
| `outputs/tables/` | `table5_critical_f.txt`, `table6_bonferroni.txt` |

## Requirements

```
matplotlib==3.9.1
networkx==3.3
numpy==1.26.4
scikit_learn==1.5.1
scipy==1.14.0
torch==2.4.0
tqdm==4.66.4
```
