import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import scipy.io.arff as arff
import matplotlib.pyplot as plt

# ---------------------------
# Dataset Definition (ECG5000)
# ---------------------------
class ECG5000(Dataset):
    def __init__(self, mode, split='train'):
        """
        mode: 'normal', 'anomaly', or 'all'. 
              'all' means do not filter any samples (both normal and anomaly).
        split: 'train' to load training data; 'test' to load test data.
        """
        assert mode in ['normal', 'anomaly', 'all']
        assert split in ['train', 'test']
        
        # Select the file based on the split.
        if split == 'train':
            file_path = 'ECG5000_TRAIN.arff'
        else:
            file_path = 'ECG5000_TEST.arff'
        
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data, columns=meta.names())
        
        # Rename the label column.
        new_columns = list(df.columns)
        new_columns[-1] = 'target'
        df.columns = new_columns
        
        # Filter samples based on mode.
        if mode == 'normal':
            df = df[df.target == b'1'].drop(labels='target', axis=1)
        elif mode == 'anomaly':
            df = df[df.target != b'1'].drop(labels='target', axis=1)
        else:  # mode == 'all'
            df = df.drop(labels='target', axis=1)
        
        # Convert DataFrame to a numpy array of type float32.
        self.X = df.astype(np.float32).to_numpy()
        
    def __getitem__(self, index):
        # Each sample is reshaped as (sequence_length, 1)
        sample = torch.from_numpy(self.X[index]).unsqueeze(-1)
        return sample
    
    def __len__(self):
        return self.X.shape[0]
    
    def get_torch_tensor(self):
        return torch.from_numpy(self.X)

class MemoryModule(nn.Module):
    def __init__(self, memory_size, hidden_size, sparsity_threshold=0.05):
        """
        memory_size: Number of memory items.
        hidden_size: Dimensionality of each memory item.
        sparsity_threshold: Threshold for rectifying the addressing vector.
        """
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        # Initialize learnable memory items.
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_size))
    
    def forward(self, z):
        """
        z: latent representation from encoder with shape (batch, hidden_size)
        Returns:
          z_hat: recombined latent representation from memory.
          q: sparse addressing vector with shape (batch, memory_size)
        """
        # Compute similarity scores between latent vector and memory items.
        sim = torch.matmul(z, self.memory.t())  # shape: (batch, memory_size)
        # Softmax to obtain addressing weights.
        q = nn.functional.softmax(sim, dim=1)
        # Rectify: subtract threshold and zero out negatives.
        q = torch.max(q - self.sparsity_threshold, torch.zeros_like(q))
        # Normalize so that each row sums to 1.
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
        # Recombine memory items.
        z_hat = torch.matmul(q, self.memory)
        return z_hat, q

# ---------------------------
# TSMAE Model Definition
# ---------------------------
class TSMAE(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, sparsity_threshold=0.05, sparsity_factor=0.001):
        """
        input_size: Dimension of each time step (e.g., 1)
        hidden_size: Dimension of the latent representation
        memory_size: Number of memory items.
        sparsity_threshold: Threshold used in the memory module.
        sparsity_factor: Weight for the sparsity penalty in the loss.
        """
        super(TSMAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.sparsity_factor = sparsity_factor
        
        # LSTM Encoder: encodes input sequence into a latent vector.
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Memory Module: extracts typical normal patterns.
        self.memory_module = MemoryModule(memory_size, hidden_size, sparsity_threshold)
        # LSTM Decoder: decodes the latent representation back to sequence.
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # Final layer to project the LSTM decoder output to the input space.
        self.output_layer = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        """
        x: Input tensor of shape (batch, seq_len, input_size)
        Returns:
          x_recon: Reconstructed sequence of shape (batch, seq_len, input_size)
          q: Sparse addressing vector from the memory module (batch, memory_size)
          z: Latent representation from the encoder (batch, hidden_size)
          z_hat: Recombined latent representation from the memory module (batch, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        # Encode input sequence.
        enc_out, (h_n, c_n) = self.encoder(x)
        z = h_n[-1]  # Use the final hidden state; shape: (batch, hidden_size)
        
        # Pass through memory module.
        z_hat, q = self.memory_module(z)
        
        # For decoding, repeat z_hat across the sequence length.
        z_hat_seq = z_hat.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(z_hat_seq)
        # Project decoder output back to input dimension.
        x_recon = self.output_layer(dec_out)
        return x_recon, q, z, z_hat

    def loss_function(self, x, x_recon, q):
        # Mean Squared Error reconstruction loss.
        rec_loss = torch.mean((x - x_recon)**2)
        # Sparsity loss to encourage a sparse addressing vector.
        sparsity_loss = torch.mean(torch.log(1 + q**2))
        loss = rec_loss + self.sparsity_factor * sparsity_loss
        return loss, rec_loss, sparsity_loss

# ---------------------------
# Training Setup
# ---------------------------
def train_model(model, dataloader, optimizer, device, num_epochs=50):
    model.to(device)
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, q, z, z_hat = model(batch)
            loss, rec_loss, sparsity_loss = model.loss_function(batch, x_recon, q)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")
    return train_losses

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    # Hyperparameters.
    input_size = 1           # Each time step has 1 feature.
    hidden_size = 10         # Latent representation dimension.
    memory_size = 20         # Number of memory items.
    sparsity_threshold = 0.05
    sparsity_factor = 0.001
    batch_size = 1
    num_epochs = 50
    learning_rate = 1e-3

    # ---------------------------
    # Training: Use the TRAIN file with both normal and anomaly samples.
    # ---------------------------
    train_dataset = ECG5000(mode='all', split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize the TSMAE model.
    model = TSMAE(input_size, hidden_size, memory_size, sparsity_threshold, sparsity_factor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check for available device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    # Train the model.
    train_losses = train_model(model, train_loader, optimizer, device, num_epochs=num_epochs)

    # Plot training loss.
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss vs Epochs")
    plt.show()

    # ---------------------------
    # Save the Model Weights
    # ---------------------------
    torch.save(model.state_dict(), 'tsmae_weights.pth')
    print("Model weights saved to 'tsmae_weights.pth'")

    # ---------------------------
    # Evaluation: Use the TEST file with both normal and anomaly samples.
    # ---------------------------
    test_dataset = ECG5000(mode='all', split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x_recon, q, z, z_hat = model(batch)
            loss, rec_loss, sparsity_loss = model.loss_function(batch, x_recon, q)
            total_test_loss += loss.item() * batch.size(0)
    avg_test_loss = total_test_loss / len(test_loader.dataset)
    print(f"Average test loss on TEST file: {avg_test_loss:.6f}")

    # ---------------------------
    # Plotting Reconstruction Comparisons (TEST file)
    # ---------------------------
    # Evaluate one normal sample from the test file.
    normal_dataset_eval = ECG5000(mode='normal', split='test')
    normal_sample = normal_dataset_eval[0].unsqueeze(0).to(device)
    with torch.no_grad():
        normal_recon, normal_q, _, _ = model(normal_sample)
    normal_series_np = normal_sample.cpu().numpy().flatten()
    normal_recon_np = normal_recon.cpu().numpy().flatten()

    # Evaluate one anomaly sample from the test file.
    anomaly_dataset_eval = ECG5000(mode='anomaly', split='test')
    anomaly_sample = anomaly_dataset_eval[0].unsqueeze(0).to(device)
    with torch.no_grad():
        anomaly_recon, anomaly_q, _, _ = model(anomaly_sample)
    anomaly_series_np = anomaly_sample.cpu().numpy().flatten()
    anomaly_recon_np = anomaly_recon.cpu().numpy().flatten()

    # Plot normal sample reconstruction.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(normal_series_np, label="Original Normal")
    plt.plot(normal_recon_np, label="Reconstructed Normal", linestyle="--")
    plt.title("Normal Sample Reconstruction (TEST file)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()

    # Plot anomaly sample reconstruction.
    plt.subplot(1, 2, 2)
    plt.plot(anomaly_series_np, label="Original Anomaly")
    plt.plot(anomaly_recon_np, label="Reconstructed Anomaly", linestyle="--")
    plt.title("Anomaly Sample Reconstruction (TEST file)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
