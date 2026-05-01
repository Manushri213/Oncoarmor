
# OncoArmor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-4.30%2B-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A deep learning framework for classifying AntiCancer Peptides (ACPs) and generating novel peptide sequences using Reinforcement Learning with protein language models.

## Features

- **Binary Classification**: State-of-the-art ACP classification using ESM2 (150M parameters)
- **De Novo Generation**: RL-based peptide generator with physicochemical property optimization
- **High Performance**: ROC-AUC of 0.90 with optimized training pipeline

## Performance Metrics

### Classification Performance
- **ROC-AUC**: 0.90
- **Accuracy**: High performance on test set

### Generation Performance
- RL training converges to ~0.8 average reward
- Generates novel peptides with high ACP probability
- Incorporates physicochemical constraints (charge, hydrophobicity)


## Configuration

Key parameters :

```python
{
    "max_len": 1024,           # Maximum sequence length
    "batch_size": 8,           # Training batch size
    "epochs": 50,              # Classifier training epochs
    "lr": 2e-5,                # Learning rate
    "model_name": "facebook/esm2_t30_150M_UR50D",
    "seed": 42,
    "output_dir": "results_rl_integration",
    "rl_epochs": 500,          # RL fine-tuning epochs
    "rl_batch_size": 64,       # RL batch size
    "vocab": "ACDEFGHIKLMNPQRSTVWY",  # Amino acid vocabulary
    "num_workers": 4,          # Parallel data loading
}
```

## Methodology

### 1. Classifier Training
- **Model**: ESM2 (Evolutionary Scale Modeling) - 150M parameters
- **Task**: Binary classification (ACP vs non-ACP)
- **Optimization**: AdamW with cosine learning rate schedule
- **Metrics**: Accuracy, MCC, ROC-AUC

### 2. Peptide Generation
- **Architecture**: GRU-based sequence generator
- **Pre-training**: Maximum Likelihood Estimation (MLE) on positive sequences
- **Fine-tuning**: Reinforcement Learning with:
  - **Reward**: Classifier probability (70%) + Physicochemical properties (30%)
  - **KL Penalty**: Maintains similarity to pre-trained distribution
  - **Entropy Bonus**: Prevents mode collapse
  - **Diversity**: Penalizes repetition within sequences

### 3. Physicochemical Rewards
- **Cationic Preference**: Rewards Arg (R), Lys (K), His (H) content > 15%
- **Hydrophobicity**: Optimal range 30-60% (I, L, M, F, V, W, Y)
- **Amphipathic Balance**: Encourages ACP-typical properties

## Results
### Dataset Statistics

#### Sequence Length Distribution
<img width="800" height="600" alt="sequence_length_frequency" src="https://github.com/user-attachments/assets/29387cb6-a39c-4260-8fde-adb2f514fd8b" />

Most sequences fall in the 10-50 amino acid range, typical for antimicrobial peptides.

#### Amino Acid Composition
<img width="1000" height="600" alt="top_10_amino_acids" src="https://github.com/user-attachments/assets/9ccef9ae-7c2e-4ebb-b440-0d235fff14f8" />


Top 10 most frequent amino acids in the dataset, showing prevalence of Leucine (L), Glycine (G), Alanine (A), and Lysine (K).

### Classification Results

#### Confusion Matrix
<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/efd0414c-346e-4cd6-8631-1dacff8975c5" />

- **True Negatives**: 700
- **True Positives**: 94
- **False Positives**: 20
- **False Negatives**: 62

#### ROC Curve
<img width="640" height="480" alt="roc_curve" src="https://github.com/user-attachments/assets/72822ae7-f9da-49d8-99d6-b864ef7d26a4" />

- **AUC Score**: 0.90
- Excellent discrimination between ACP and non-ACP sequences

### Embedding Visualizations

#### t-SNE Projection
<img width="1000" height="800" alt="tsne_plot" src="https://github.com/user-attachments/assets/cb293844-a7ac-4cc4-b7fa-01799f6396bf" />

The t-SNE visualization shows the 2D projection of protein embeddings from the ESM2 model. The separation between classes (0: non-ACP, 1: ACP) demonstrates the model's ability to learn discriminative features.

#### UMAP Projection
<img width="1000" height="800" alt="umap_plot" src="https://github.com/user-attachments/assets/8dc8590c-0a75-4f36-accb-641f7afebbae" />


UMAP provides an alternative dimensionality reduction showing clearer clustering of ACP sequences (class 1) in distinct regions of the embedding space, indicating strong feature representation.


### Generation Results

#### RL Training Progress
<img width="1000" height="500" alt="rl_training_metrics" src="https://github.com/user-attachments/assets/bd9dc409-3dc0-4423-9fe9-ff1c7a35f46b" />


- **Left**: Average reward (ACP probability) increases steadily from 0.2 to ~0.8
- **Right**: Generator loss stabilizes after initial training phase

## License

Distributed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work uses the **(https://github.com/TearsWaiting/ACPred-LAF)** benchmark dataset, a gold‑standard resource for anticancer peptide research. We thank Wenjia He, Yu Wang, Lei Cui, Ran Su, and Leyi Wei for making this data publicly available.

**Please cite the following reference when using the dataset:**

He, W., Wang, Y., Cui, L., Su, R., & Wei, L. (2021). Learning embedding features based on multisense-scaled attention architecture to improve the predictive performance of anticancer peptides. *Bioinformatics*, 37(24), 4684-4693. [https://doi.org/10.1093/bioinformatics/btab560](https://doi.org/10.1093/bioinformatics/btab560)

**Resources:**
- ACPred-LAF GitHub: [https://github.com/TearsWaiting/ACPred-LAF](https://github.com/TearsWaiting/ACPred-LAF)
- ACPred-LAF Web Server: [http://server.malab.cn/ACPred-LAF](http://server.malab.cn/ACPred-LAF)

We also thank the developers of Hugging Face Transformers, Facebook Research ESM team, and the open‑source community for their essential tools.

## Authors

- [Manushri Nerlekar](https://github.com/Manushri213)
- [Surya Srikar](https://github.com/kssrikar4)
