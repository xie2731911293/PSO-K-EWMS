# -*- coding: utf-8 -*-
"""CVAE-based gene screening for the grape double-cropping data.

Conditional Variational Autoencoder (CVAE) used to filter genes whose
expression profiles are dominated by noise, before clustering.

Usage:
    python CVAE.py --input meta_gene_common.xlsx [options]

The input spreadsheet must contain the columns
    Sample, Season, GrowthStage, cultivar
followed by one column per gene (the expression values).
The one-hot encodings of the categorical covariates
`Season`, `GrowthStage`, and `cultivar` are used as conditioning variables.

All defaults reproduce the original experimental setting (seed=42,
test_size=0.2, latent_dim=20, epochs=100, beta=1.0, top 10% percentile).
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(input_dim + condition_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_mean = nn.Linear(256, latent_dim)
        self.fc3_logvar = nn.Linear(256, latent_dim)
        # decoder
        self.fc4 = nn.Linear(latent_dim + condition_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, input_dim)

    def encode(self, x, c):
        h = torch.relu(self.fc1(torch.cat([x, c], dim=-1)))
        h = torch.relu(self.fc2(h))
        return self.fc3_mean(h), self.fc3_logvar(h)

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z, c):
        h = torch.relu(self.fc4(torch.cat([z, c], dim=-1)))
        h = torch.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x, c):
        z_mean, z_logvar = self.encode(x, c)
        z = self.reparameterize(z_mean, z_logvar)
        return self.decode(z, c), z_mean, z_logvar


def loss_function(reconstructed_x, x, z_mean, z_logvar, beta=1.0):
    reconstruction_loss = nn.MSELoss()(reconstructed_x, x)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return reconstruction_loss + beta * kl_loss


def main():
    ap = argparse.ArgumentParser(description="CVAE-based gene screening")
    ap.add_argument("--input", default="meta_gene_common.xlsx",
                    help="path to the gene-expression spreadsheet")
    ap.add_argument("--out", default=None,
                    help="optional csv path to save the selected gene names")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--latent_dim", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="retain genes with reconstruction error below this percentile")
    args = ap.parse_args()

    # ---------- data loading ----------
    data = pd.read_excel(args.input)
    gene_data = data.drop(columns=["Sample", "Season", "GrowthStage", "cultivar"])
    gene_names = gene_data.columns
    cov = data[["Season", "GrowthStage", "cultivar"]]
    cond = pd.get_dummies(cov).values.astype(np.float32)              # (n, condition_dim)

    scaler = StandardScaler()
    gene_scaled = scaler.fit_transform(gene_data)

    # design matrix = scaled genes + one-hot condition
    X = np.hstack([gene_scaled, cond])
    condition_dim = cond.shape[1]
    input_dim = gene_data.shape[1]

    X_train, X_test = train_test_split(X, test_size=args.test_size, random_state=args.seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # ---------- model ----------
    torch.manual_seed(args.seed)
    model = CVAE(input_dim, args.latent_dim, condition_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------- training ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        recon, z_mean, z_logvar = model(X_train_t, X_train_t[:, -condition_dim:])
        loss = loss_function(recon, X_train_t, z_mean, z_logvar, beta=args.beta)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{args.epochs}], Loss: {loss.item():.4f}")

    # ---------- evaluation & screening ----------
    model.eval()
    with torch.no_grad():
        recon_test, _, _ = model(X_test_t, X_test_t[:, -condition_dim:])
        errors = torch.mean((recon_test - X_test_t) ** 2, dim=0).numpy()[:input_dim]

    threshold = np.percentile(errors, args.percentile)
    important_idx = np.where(errors <= threshold)[0]
    important_genes = gene_names[important_idx]

    print(f"Selected {len(important_genes)} genes "
          f"(reconstruction error <= {args.percentile:.1f}th percentile):")
    for g in important_genes:
        print(" ", g)

    if args.out:
        pd.DataFrame({"gene": important_genes}).to_csv(args.out, index=False)
        print(f"Saved selected genes to {args.out}")


if __name__ == "__main__":
    main()
