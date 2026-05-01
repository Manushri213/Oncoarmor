import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
    logging as hf_logging
)
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from tqdm import tqdm
import glob
import shutil
from collections import Counter

# --- Aesthetics ---
BROWN_DARK = "#5D4037"
BROWN_LIGHT = "#D7CCC8"

# --- Optimizations ---
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True         # Auto-tuner for convolution algos

hf_logging.set_verbosity_info()

CONFIG = {
    "max_len": 1024,       # Capable of 1024, but Dynamic Padding will optimize actual usage
    "batch_size": 8,       # Reduced slightly to accommodate larger 35M model comfortably in VRAM
    "epochs": 50,
    "lr": 2e-5,
    "model_name": "facebook/esm2_t30_150M_UR50D", # Upgraded to 35M Parameter Model
    "seed": 42,
    "output_dir": "results_rl_integration",
    "rl_epochs": 500,
    "rl_batch_size": 64,   # Larger batch size for RL throughput
    "vocab": "ACDEFGHIKLMNPQRSTVWY",
    "num_workers": min(4, multiprocessing.cpu_count()), # Parallel data loading
}

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

class ACPDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len):
        self.sequences = [str(s) for s in sequences]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Precompute lengths for group_by_length optimization
        self.lengths = [len(s) for s in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    try:
        roc_auc = roc_auc_score(labels, probs)
    except:
        roc_auc = 0.0
    return {'accuracy': acc, 'mcc': mcc, 'roc_auc': roc_auc}

class PeptideGenerator(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256, vocab_size=21, max_len=50):
        super(PeptideGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.rnn(embed, hidden)
        logits = self.fc(output)
        return logits, hidden

    def sample(self, num_samples, start_token=0, temperature=1.0):
        device = next(self.parameters()).device
        inputs = torch.tensor([[start_token]] * num_samples, device=device)
        hidden = None
        generated_seqs = []
        log_probs = []

        for _ in range(self.max_len):
            logits, hidden = self.forward(inputs, hidden)
            # Apply temperature scaling
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample()
            log_prob = dist.log_prob(next_token)
            
            generated_seqs.append(next_token.unsqueeze(1))
            log_probs.append(log_prob.unsqueeze(1))
            inputs = next_token.unsqueeze(1)
        
        generated_seqs = torch.cat(generated_seqs, dim=1)
        log_probs = torch.cat(log_probs, dim=1)
        return generated_seqs, log_probs

class GeneratorDataset(Dataset):
    def __init__(self, sequences, vocab, max_len):
        self.vocab_map = {c: i+1 for i, c in enumerate(vocab)}
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Truncate if needed
        seq = seq[:self.max_len-1]
        tokens = [0] + [self.vocab_map.get(c, 0) for c in seq] # 0 as start token
        padding = [0] * (self.max_len - len(tokens))
        input_tokens = torch.tensor(tokens + padding, dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:] + [0] + padding, dtype=torch.long)
        return input_tokens, target_tokens

def pretrain_generator(generator, sequences, vocab, device, epochs=20):
    print("--- Pre-training Generator (MLE) ---")
    dataset = GeneratorDataset(sequences, vocab, CONFIG["max_len"] if CONFIG["max_len"] < 50 else 50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(generator.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    generator.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = generator(inputs)
            loss = criterion(logits.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# --- Improved RL Trainer to prevent Mode Collapse ---
class RLTrainer:
    def __init__(self, generator, ref_generator, reward_model, tokenizer, device, vocab, entropy_coef=0.01, kl_coef=0.05):
        self.generator = generator
        self.ref_generator = ref_generator # Frozen pre-trained model for KL penalty
        if self.ref_generator:
            self.ref_generator.eval()
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab = vocab
        self.idx_to_char = {i+1: c for i, c in enumerate(vocab)}
        self.optimizer = optim.Adam(self.generator.parameters(), lr=1e-4) 
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.baseline = 0.5

    def decode_seq(self, seq_indices):
        seqs = []
        for seq in seq_indices:
            s = "".join([self.idx_to_char.get(idx.item(), "") for idx in seq if idx.item() != 0])
            seqs.append(s)
        return seqs

    def calculate_physicochemical_reward(self, seq):
        if not seq: return 0.0
        # Cationic preference: R (Arg), K (Lys), H (His)
        cationic_aas = "RKH"
        charge_score = sum(seq.count(aa) for aa in cationic_aas) / len(seq)
        
        # Hydrophobicity estimate (simple)
        hydrophobic_aas = "AILMFVWY"
        hydro_score = sum(seq.count(aa) for aa in hydrophobic_aas) / len(seq)
        
        # ACPs are often cationic and amphipathic (balanced hydrophobicity)
        # Reward charge > 0.15 and hydrophobicity between 0.3 and 0.6
        reward = 0.0
        if charge_score > 0.15: reward += 0.2
        if 0.3 < hydro_score < 0.6: reward += 0.2
        return reward

    def get_reward(self, sequences):
        self.reward_model.eval()
        rewards = torch.zeros(len(sequences), device=self.device)
        
        valid_indices = [i for i, seq in enumerate(sequences) if len(seq) >= 6]
        if not valid_indices:
            return rewards

        valid_seqs = [sequences[i] for i in valid_indices]
        
        inputs = self.tokenizer(
            valid_seqs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=CONFIG["max_len"]
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[:, 1]
        
        # Diversity within batch (Levenshtein-based penalty is slow, using set-based uniqueness)
        unique_seqs = set(sequences)
        diversity_multiplier = len(unique_seqs) / len(sequences)

        for i, idx in enumerate(valid_indices):
            seq = valid_seqs[i]
            prob_score = probs[i].item()
            
            # Penalize repetition within sequence
            counts = {aa: seq.count(aa) for aa in set(seq)}
            max_repeat = max(counts.values()) / len(seq) if seq else 1.0
            repetition_penalty = 0.5 if max_repeat > 0.3 else 0.0
            
            physico_reward = self.calculate_physicochemical_reward(seq)
            
            # Final Reward
            final_reward = (prob_score * 0.7) + (physico_reward * 0.3) - repetition_penalty
            rewards[idx] = final_reward * diversity_multiplier
            
        return rewards

    def train_step(self, batch_size):
        self.generator.train()
        self.optimizer.zero_grad()
        
        # Sample with temperature for exploration
        seq_indices, log_probs = self.generator.sample(batch_size, start_token=0, temperature=1.2)
        decoded_seqs = self.decode_seq(seq_indices)
        rewards = self.get_reward(decoded_seqs)
        
        # KL Divergence Penalty (keep model close to pre-trained protein distribution)
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.ref_generator:
            with torch.no_grad():
                # Re-run ref_generator on the sampled indices to get its log_probs
                ref_logits, _ = self.ref_generator(seq_indices)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                # Gather log probs for the actual tokens sampled
                # seq_indices: [batch, seq_len] -> targets for gather
                target_indices = seq_indices.unsqueeze(-1)
                ref_sampled_log_probs = ref_log_probs.gather(2, target_indices).squeeze(-1)
                
            kl_loss = (log_probs - ref_sampled_log_probs).mean()

        entropy = -log_probs.mean() 
        advantage = rewards - self.baseline
        self.baseline = 0.9 * self.baseline + 0.1 * rewards.mean().item()
        
        pg_loss = -(log_probs.sum(dim=1) * advantage).mean()
        loss = pg_loss - (self.entropy_coef * entropy) + (self.kl_coef * kl_loss)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item(), rewards.mean().item()

def train_classifier(df_train, df_test, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=2)
    
    train_dataset = ACPDataset(df_train['Seq'].tolist(), df_train['Label'].tolist(), tokenizer, CONFIG["max_len"])
    test_dataset = ACPDataset(df_test['Seq'].tolist(), df_test['Label'].tolist(), tokenizer, CONFIG["max_len"])
    
    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG["lr"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"] * 2,
        num_train_epochs=CONFIG["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        save_total_limit=1,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0, # Set to 0 for stability on Windows
        dataloader_pin_memory=True,
        group_by_length=False,
        disable_tqdm=False, # Ensure progress bar is visible
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8), # pad to multiple of 8 for tensor core efficiency
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    # Delete all checkpoints
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Deleted checkpoints folder: {checkpoint_dir}")

    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # --- Fix Checkpoint Keys (Gamma/Beta -> Weight/Bias) ---
    def fix_checkpoint_keys(model_path):
        from safetensors.torch import load_file, save_file
        print(f"Checking keys in {model_path}...")
        safe_path = os.path.join(model_path, "model.safetensors")
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(safe_path):
            state_dict = load_file(safe_path)
            path_to_save = safe_path
            is_safe = True
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            path_to_save = bin_path
            is_safe = False
        else:
            print("No model file found to fix.")
            return

        new_state_dict = {}
        fixed = False
        for k, v in state_dict.items():
            if "LayerNorm.gamma" in k:
                new_k = k.replace("LayerNorm.gamma", "LayerNorm.weight")
                new_state_dict[new_k] = v.clone()
                fixed = True
            elif "LayerNorm.beta" in k:
                new_k = k.replace("LayerNorm.beta", "LayerNorm.bias")
                new_state_dict[new_k] = v.clone()
                fixed = True
            else:
                new_state_dict[k] = v.clone()
        
        # Explicitly delete the old state_dict to release file handles if any
        del state_dict
        import gc
        gc.collect()

        if fixed:
            print(f"Fixed LayerNorm keys in {path_to_save}")
            if is_safe:
                # Save to a temp file first to avoid locking issues on Windows
                temp_path = path_to_save + ".tmp"
                try:
                    save_file(new_state_dict, temp_path)
                    # Replace the original file
                    if os.path.exists(path_to_save):
                        os.remove(path_to_save)
                    os.rename(temp_path, path_to_save)
                except Exception as e:
                    print(f"Error saving fixed model: {e}")
            else:
                torch.save(new_state_dict, path_to_save)
        else:
            print("No keys needed fixing.")

    fix_checkpoint_keys(os.path.join(output_dir, "final_model"))

    return trainer, model, tokenizer, test_dataset

def plot_results(trainer, test_dataset, output_dir):
    from matplotlib.colors import LinearSegmentedColormap
    brown_cmap = LinearSegmentedColormap.from_list("browns", [BROWN_LIGHT, BROWN_DARK])

    preds_output = trainer.predict(test_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_prob = F.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()[:, 1]
    y_true = preds_output.label_ids

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=brown_cmap)
    plt.title('Confusion Matrix', color=BROWN_DARK, fontweight='bold')
    plt.ylabel('True', color=BROWN_DARK)
    plt.xlabel('Predicted', color=BROWN_DARK)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color=BROWN_DARK, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color=BROWN_LIGHT, linestyle='--')
    plt.xlabel('False Positive Rate', color=BROWN_DARK)
    plt.ylabel('True Positive Rate', color=BROWN_DARK)
    plt.title('ROC Curve', color=BROWN_DARK, fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_rl_progress(rewards, losses, output_dir):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, color=BROWN_DARK)
    plt.title("Average Reward (ACP Probability)", color=BROWN_DARK, fontweight='bold')
    plt.xlabel("Batch", color=BROWN_DARK)
    plt.ylabel("Reward", color=BROWN_DARK)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, color=BROWN_DARK)
    plt.title("Generator Loss", color=BROWN_DARK, fontweight='bold')
    plt.xlabel("Batch", color=BROWN_DARK)
    plt.ylabel("Loss", color=BROWN_DARK)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rl_training_metrics.png'))
    plt.close()

def plot_embeddings(model, tokenizer, test_dataset, output_dir):
    print("Generating embeddings for UMAP/t-SNE...")
    model.eval()
    device = next(model.parameters()).device
    
    # Create dataloader with padding to handle variable sequence lengths
    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].cpu().numpy()
            
            # Forward pass through the base ESM model
            # For EsmForSequenceClassification, access model.esm
            outputs = model.esm(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get CLS token embedding (first token)
            # shape: (batch_size, seq_len, hidden_dim)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            embeddings.append(cls_embeddings.cpu().numpy())
            labels.append(batch_labels)
            
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=labels, palette=[BROWN_LIGHT, BROWN_DARK], alpha=0.7)
    plt.title('t-SNE of Protein Embeddings', color=BROWN_DARK, fontweight='bold')
    plt.xlabel('t-SNE 1', color=BROWN_DARK)
    plt.ylabel('t-SNE 2', color=BROWN_DARK)
    plt.savefig(os.path.join(output_dir, 'tsne_plot.png'))
    plt.close()
    
    # UMAP
    try:
        import umap
        print("Running UMAP...")
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=labels, palette=[BROWN_LIGHT, BROWN_DARK], alpha=0.7)
        plt.title('UMAP of Protein Embeddings', color=BROWN_DARK, fontweight='bold')
        plt.xlabel('UMAP 1', color=BROWN_DARK)
        plt.ylabel('UMAP 2', color=BROWN_DARK)
        plt.savefig(os.path.join(output_dir, 'umap_plot.png'))
        plt.close()
    except ImportError:
        print("UMAP library not found. Skipping UMAP plot. (pip install umap-learn)")

def plot_dataset_stats(df, output_dir):
    print("Generating dataset statistics plots...")
    # Sequence Length Frequency
    lengths = df['Seq'].str.len()
    plt.figure(figsize=(8, 6))
    sns.histplot(lengths, bins=30, color=BROWN_DARK, kde=True)
    plt.title('Sequence Length Frequency', color=BROWN_DARK, fontsize=14, fontweight='bold')
    plt.xlabel('Length', color=BROWN_DARK)
    plt.ylabel('Frequency', color=BROWN_DARK)
    plt.savefig(os.path.join(output_dir, 'sequence_length_frequency.png'))
    plt.close()

    # Top 10 Amino Acids
    all_seqs = "".join(df['Seq'].tolist())
    aa_counts = Counter(all_seqs)
    top_10_aa = aa_counts.most_common(10)
    aa_labels, aa_values = zip(*top_10_aa)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(aa_labels), y=list(aa_values), hue=list(aa_labels), palette=[BROWN_DARK, BROWN_LIGHT] * 5, legend=False)
    plt.title('Top 10 Amino Acids in Dataset', color=BROWN_DARK, fontsize=14, fontweight='bold')
    plt.xlabel('Amino Acid', color=BROWN_DARK)
    plt.ylabel('Count', color=BROWN_DARK)
    plt.savefig(os.path.join(output_dir, 'top_10_amino_acids.png'))
    plt.close()

def main():
    set_global_seed(CONFIG["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load all datasets
    dataset_dir = "dataset/ACPred-LAF"
    all_csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    
    print(f"Found {len(all_csv_files)} CSV files in {dataset_dir}")
    
    def load_all_csvs(files):
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip()
                if 'Seq' in df.columns and 'Label' in df.columns:
                    # Filter out non-standard amino acids if any, and convert to uppercase
                    df['Seq'] = df['Seq'].astype(str).str.upper().str.strip()
                    dfs.append(df[['Seq', 'Label']])
            except Exception as e:
                print(f"Error loading {f}: {e}")
        if not dfs:
            return pd.DataFrame(columns=['Seq', 'Label'])
        
        combined_df = pd.concat(dfs, ignore_index=True)
        # Drop duplicates based on Sequence to ensure no leakage if we split later
        combined_df = combined_df.drop_duplicates(subset=['Seq']).reset_index(drop=True)
        return combined_df

    full_df = load_all_csvs(all_csv_files)
    
    # Filter out empty or invalid sequences
    full_df = full_df[full_df['Seq'].str.len() > 0]
    
    print(f"Total unique samples across all datasets: {len(full_df)}")

    if len(full_df) == 0:
        print("No data found. Exiting.")
        return

    plot_dataset_stats(full_df, CONFIG["output_dir"])

    # Split into train and test
    df_train, df_test = train_test_split(full_df, test_size=0.15, random_state=CONFIG["seed"], stratify=full_df['Label'])
    
    print(f"Training on {len(df_train)} samples, testing on {len(df_test)} samples.")

    trainer, classifier, tokenizer, test_dataset = train_classifier(df_train, df_test, CONFIG["output_dir"])
    plot_results(trainer, test_dataset, CONFIG["output_dir"])
    plot_embeddings(classifier, tokenizer, test_dataset, CONFIG["output_dir"])

    print("Starting Optimized De Novo Generation Pipeline...")
    
    # 1. Generator Pre-training (MLE)
    # Use positive sequences for pre-training to learn "ACP grammar"
    pos_sequences = df_train[df_train['Label'] == 1]['Seq'].tolist()
    generator = PeptideGenerator(vocab_size=len(CONFIG["vocab"])+1).to(device)
    pretrain_generator(generator, pos_sequences, CONFIG["vocab"], device, epochs=50)
    
    # Save pre-trained base
    torch.save(generator.state_dict(), os.path.join(CONFIG["output_dir"], "generator_pretrain.pth"))
    
    # 2. RL Fine-tuning
    print("Starting Reinforcement Learning Fine-tuning...")
    # Create a reference generator (frozen) for KL penalty
    ref_generator = PeptideGenerator(vocab_size=len(CONFIG["vocab"])+1).to(device)
    ref_generator.load_state_dict(generator.state_dict())
    
    rl_trainer = RLTrainer(generator, ref_generator, classifier, tokenizer, device, CONFIG["vocab"])
    
    rl_rewards = []
    rl_losses = []
    
    pbar = tqdm(range(CONFIG["rl_epochs"]), desc="RL Training")
    for epoch in pbar:
        loss, avg_reward = rl_trainer.train_step(CONFIG["rl_batch_size"])
        rl_rewards.append(avg_reward)
        rl_losses.append(loss)
        pbar.set_postfix({"reward": f"{avg_reward:.4f}", "loss": f"{loss:.4f}"})

    plot_rl_progress(rl_rewards, rl_losses, CONFIG["output_dir"])
    
    # Save the Generator
    torch.save(generator.state_dict(), os.path.join(CONFIG["output_dir"], "generator.pth"))
    
    print("Generating top candidate peptides...")
    generator.eval()
    with torch.no_grad():
        # Generate more candidates to find best ones
        seq_indices, _ = generator.sample(50)
        candidates = rl_trainer.decode_seq(seq_indices)
        rewards = rl_trainer.get_reward(candidates)
        
    # Sort by reward
    sorted_indices = torch.argsort(rewards, descending=True)
    top_candidates = [(candidates[i], rewards[i].item()) for i in sorted_indices[:20]]

    with open(os.path.join(CONFIG["output_dir"], "generated_peptides.txt"), "w") as f:
        f.write("Sequence,Predicted_ACP_Probability\n")
        for s, r in top_candidates:
            f.write(f"{s},{r:.6f}\n")
    
    print(f"Process Complete. Results in {CONFIG['output_dir']}")

if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows support
    main()