#%%

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def sample_2d_data(num_samples=2000):
    """Sample points from a mixture of Gaussians in 2D."""
    # Mixture of Gaussians:
    centers = [
        (0, 0),
        (4, 0),
        (0, 4),
        (-4, 0),
        (0, -4)
    ]
    samples_per_center = num_samples // len(centers)
    data = []
    for (cx, cy) in centers:
        x = np.random.randn(samples_per_center, 2) * 0.5
        x += np.array([cx, cy])
        data.append(x)
    data = np.vstack(data).astype(np.float32)
    np.random.shuffle(data)
    return data

class RealNVPCouplingLayer(nn.Module):
    def __init__(self, in_features, hidden_features, mask_type="odd"):
        """
        Args:
            in_features: Number of input features (e.g., 2 for 2D).
            hidden_features: Number of hidden units in the MLP.
            mask_type: "odd" or "even" to choose which dimensions to transform.
        """
        super().__init__()
        self.mask_type = mask_type
        # Simple MLP to predict scale and translation from part of x
        self.net = nn.Sequential(
            nn.Linear(in_features // 2, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features)  # outputs [scale, translation]
        )

    def forward(self, x):
        """
        Forward pass (x -> y). Returns y, log_det_jacobian.
        """
        x_a, x_b = self._split_x(x)  # x_a is the part used for MLP, x_b is transformed
        # Predict [scale, translation] from x_a
        st = self.net(x_a)
        s, t = st.chunk(2, dim=1)  # split into scale and translation
        # Apply scale & translation to x_b
        y_b = x_b * torch.exp(s) + t
        # Reconstruct full output
        y = self._combine_x(x_a, y_b)
        # log determinant is sum of s
        log_det_jacobian = s.sum(dim=1)
        return y, log_det_jacobian

    def inverse(self, y):
        """
        Inverse pass (y -> x). Returns x, log_det_jacobian.
        """
        y_a, y_b = self._split_x(y)
        st = self.net(y_a)
        s, t = st.chunk(2, dim=1)
        # Invert scale & translation
        x_b = (y_b - t) * torch.exp(-s)
        # Combine
        x = self._combine_x(y_a, x_b)
        # log determinant is -sum of s
        log_det_jacobian = -s.sum(dim=1)
        return x, log_det_jacobian

    def _split_x(self, x):
        """
        Split x into two halves based on mask_type.
        For 2D: x is shape [batch, 2]. We'll select one dimension
        for x_a and the other for x_b.
        """
        if self.mask_type == "odd":
            return x[:, 0:1], x[:, 1:2]  # x_a is dim0, x_b is dim1
        else:  # "even"
            return x[:, 1:2], x[:, 0:1]

    def _combine_x(self, x_a, x_b):
        """
        Recombine x_a and x_b in the correct order depending on mask_type.
        """
        if self.mask_type == "odd":
            return torch.cat([x_a, x_b], dim=1)
        else:
            return torch.cat([x_b, x_a], dim=1)

class RealNVP(nn.Module):
    def __init__(self, in_features=2, hidden_features=64, num_coupling_layers=4):
        super().__init__()
        layers = []
        mask_types = ["odd", "even"] * (num_coupling_layers // 2)
        for i in range(num_coupling_layers):
            layers.append(
                RealNVPCouplingLayer(
                    in_features=in_features,
                    hidden_features=hidden_features,
                    mask_type=mask_types[i]
                )
            )
        self.coupling_layers = nn.ModuleList(layers)

        # Base distribution is standard normal in 2D
        self.register_buffer("base_mean", torch.zeros(in_features))
        self.register_buffer("base_log_std", torch.zeros(in_features))

    def forward(self, x):
        """
        Forward pass: x -> z. Returns z, sum_log_det_jacobian.
        We interpret z as samples drawn from the base distribution
        (if x is real data).
        """
        log_det_jacobian_total = 0
        out = x
        for layer in self.coupling_layers:
            out, log_det_jac = layer(out)
            log_det_jacobian_total += log_det_jac
        return out, log_det_jacobian_total

    def inverse(self, z):
        """
        Inverse pass: z -> x. Useful for sampling from the model.
        """
        log_det_jacobian_total = 0
        out = z
        # Reverse order for inverse
        for layer in reversed(self.coupling_layers):
            out, log_det_jac = layer.inverse(out)
            log_det_jacobian_total += log_det_jac
        return out, log_det_jacobian_total

    def log_prob(self, x):
        """
        Compute log p(x).
        - First transform x -> z using the flow
        - Then compute log p(z) under the base distribution
        - Then add the log determinant of the Jacobian
        """
        z, log_det_jac = self.forward(x)
        # log p(z) under standard normal
        log_p_z = -0.5 * torch.sum(z**2, dim=1)  # ignoring constants for simplicity
        # Or do the full Gaussian if you want to include constants:
        # log_p_z = -0.5 * torch.sum(((z - self.base_mean) / self.base_log_std.exp())**2, dim=1)
        return log_p_z + log_det_jac

    def sample(self, num_samples):
        """
        Sample from the flow by sampling z from the base distribution,
        then transforming z -> x using inverse().
        """
        z = torch.randn(num_samples, 2).to(self.base_mean.device)
        x, _ = self.inverse(z)
        return x

def train_flow(model, data, num_epochs=1000, batch_size=256, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            x_batch = batch[0].to(device)
            # Compute negative log likelihood
            log_prob = model.log_prob(x_batch)
            loss = -log_prob.mean()  # want to maximize log_prob, so minimize negative log_prob

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)

        avg_loss = total_loss / len(data)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # 1. Create a toy dataset
    data_np = sample_2d_data(num_samples=2000)
    data_torch = torch.from_numpy(data_np)
    # 2. Instantiate the RealNVP model
    model = RealNVP(in_features=2, hidden_features=64, num_coupling_layers=4)
    print(model)
    # 3. Train the Normalizing Flow
    train_flow(model, data_torch, num_epochs=1000, batch_size=256, lr=1e-3, device="mps")
    # 4. Sample from the learned flow
    model.eval()
    samples = model.sample(num_samples=1000).detach().cpu().numpy()

    # 5. Plot the learned distribution vs. original data
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Real Data")
    plt.scatter(data_np[:, 0], data_np[:, 1], alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.title("Flow Samples")
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, color='orange')

    plt.show()
