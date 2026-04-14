import os
import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm

# ---- Diffusion-TS Imports ----
from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.context_fid import Context_FID
from Utils.discriminative_metric import discriminative_score_metrics

# ==========================================================
# 1. FIXED-POINT SOLVER (PICARD ITERATION)
# ==========================================================
def fixed_point_solver(f, x0, max_iter=100, tol=1e-6):
    """
    Pure Fixed-Point Iteration. 
    Because IDL uses Spectral Normalization, the function is a contraction mapping.
    This guarantees convergence and is immune to the matrix singularities that break Broyden.
    """
    x = x0.clone()
    for _ in range(max_iter):
        x_next = f(x)
        # Check maximum absolute residual across the entire batch
        res = torch.max(torch.abs(x_next - x))
        
        if res < tol:
            break
        x = x_next
        
    return x

# ==========================================================
# 2. BLACK-BOX ROOT FINDER (LIPSWISH ACTIVATION)
# ==========================================================
class ImplicitRootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, U, b_x):
        batch = U.shape[1]
        n = A.shape[0]
        X_init = torch.zeros(n, batch, device=U.device)
        
        def f(X):
            Z_eq = A @ X + B @ U + b_x
            # Picard Iteration requires the Next State (Contraction Mapping)
            return (Z_eq * torch.sigmoid(Z_eq)) / 1.1
            
        with torch.no_grad():
            X = fixed_point_solver(f, X_init, max_iter=300, tol=1e-7) 
                
        ctx.save_for_backward(A, B, X, U, b_x)
        return X

    @staticmethod
    def backward(ctx, grad_X):
        A, B, X, U, b_x = ctx.saved_tensors
        with torch.no_grad():
            Z_eq = A @ X + B @ U + b_x
            S = torch.sigmoid(Z_eq)
            
            Phi = (S + Z_eq * S * (1.0 - S)) / 1.1 
            
            def f_adj(Y):
                # The adjoint system is also a contraction mapping
                return grad_X + A.T @ (Phi * Y)
                
            Y_init = torch.zeros_like(grad_X)
            Y = fixed_point_solver(f_adj, Y_init, max_iter=300, tol=1e-10)
                
            Phi_Y = Phi * Y
            grad_A, grad_B = Phi_Y @ X.T, Phi_Y @ U.T
            grad_U, grad_b_x = B.T @ Phi_Y, Phi_Y.sum(dim=1, keepdim=True)
            
        return grad_A, grad_B, grad_U, grad_b_x

# ==========================================================
# 3. PURE IDL BLOCK (Sequence-Level)
# ==========================================================
class PureIDLFlowBlock(nn.Module):
    def __init__(self, p, n=128):
        super().__init__()
        self.p, self.n = p, n
        
        # [NEW FIX 1] Added n_power_iterations=5 to ensure strict Lipschitz bound
        self.A_layer = nn.utils.spectral_norm(nn.Linear(n, n, bias=False), n_power_iterations=20)
        self.B_layer = nn.utils.spectral_norm(nn.Linear(p, n, bias=False), n_power_iterations=20)
        self.C_layer = nn.utils.spectral_norm(nn.Linear(n, p, bias=False), n_power_iterations=20)
        
        self.b_x = nn.Parameter(torch.zeros(n, 1))
        self.b_y = nn.Parameter(torch.zeros(p, 1))
        self.register_buffer("D", torch.eye(p))
        
        self.p_A = nn.Parameter(torch.tensor(0.0))
        self.p_B = nn.Parameter(torch.tensor(0.0))
        self.p_C = nn.Parameter(torch.tensor(0.0))
        self.kappa = 0.95
        
    def forward(self, u):
        batch = u.shape[0]
        U = u.T 
        
        _ = self.A_layer(torch.zeros(1, self.n, device=u.device))
        _ = self.B_layer(torch.zeros(1, self.p, device=u.device))
        _ = self.C_layer(torch.zeros(1, self.n, device=u.device))
        
        alpha, beta, gamma = torch.sigmoid(self.p_A), torch.exp(self.p_B), torch.sigmoid(self.p_C)
        A = self.A_layer.weight * (self.kappa * alpha)
        B = self.B_layer.weight * beta
        C = self.C_layer.weight * ((self.kappa * (1.0 - alpha) / beta) * gamma)
        
        X = ImplicitRootFinder.apply(A, B, U, self.b_x)
        
        Y = C @ X + self.D @ U + self.b_y
        y = Y.T 
        
        Z_eq = A @ X + B @ U + self.b_x
        S = torch.sigmoid(Z_eq)
        Phi = (S + Z_eq * S * (1.0 - S)) / 1.1
        Phi_batch = Phi.T.unsqueeze(-1)
        
        I = torch.eye(self.n, device=u.device).unsqueeze(0).expand(batch, self.n, self.n)
        A_eff = I - Phi_batch * A.unsqueeze(0)
        A_eff = A_eff + torch.eye(self.n, device=u.device).unsqueeze(0) * 1e-4 
        
        V = torch.linalg.solve(A_eff, Phi_batch * B.unsqueeze(0))
        
        J = C.unsqueeze(0).expand(batch, self.p, self.n) @ V + self.D.unsqueeze(0).expand(batch, self.p, self.p)
        
        # [NEW FIX 2] Replaced torch.det with slogdet to prevent FP32 underflow
        sign, logabsdet = torch.linalg.slogdet(J)
        log_det_J = logabsdet
        
        return y, log_det_J

    @torch.no_grad()
    def inverse(self, y):
        batch = y.shape[0]
        Y = y.T
        
        _ = self.A_layer(torch.zeros(1, self.n, device=y.device))
        _ = self.B_layer(torch.zeros(1, self.p, device=y.device))
        _ = self.C_layer(torch.zeros(1, self.n, device=y.device))
        
        alpha, beta, gamma = torch.sigmoid(self.p_A), torch.exp(self.p_B), torch.sigmoid(self.p_C)
        A = self.A_layer.weight * (self.kappa * alpha)
        B = self.B_layer.weight * beta
        C = self.C_layer.weight * ((self.kappa * (1.0 - alpha) / beta) * gamma)
        
        A_tilde, B_tilde = A - B @ C, B 
        b_tilde = self.b_x - B @ self.b_y 
        
        def f_inv(X):
            Z_eq = A_tilde @ X + B_tilde @ Y + b_tilde
            # Picard Iteration requires the Next State
            return (Z_eq * torch.sigmoid(Z_eq)) / 1.1
            
        X = fixed_point_solver(f_inv, torch.zeros(self.n, batch, device=y.device), max_iter=300, tol=1e-7)
        return (Y - C @ X - self.b_y).T

# ==========================================================
# 4. JOINT SEQUENCE IDL MODEL
# ==========================================================
class SequenceIDL(nn.Module):
    def __init__(self, seq_len, dim, n_flow=128, num_layers=6):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.p = seq_len * dim  
        
        self.flows = nn.ModuleList([PureIDLFlowBlock(self.p, n_flow) for _ in range(num_layers)])
        
    def forward(self, x):
        batch = x.shape[0]
        z = x.reshape(batch, self.p)
        
        total_log_det = 0
        for flow in self.flows:
            z, log_det = flow(z)
            total_log_det += log_det
            
        return z, total_log_det

    @torch.no_grad()
    def generate(self, batch_size, device):
        self.eval()
        z = torch.randn(batch_size, self.p, device=device)
        
        for flow in reversed(self.flows):
            z = flow.inverse(z)
            
        return z.view(batch_size, self.seq_len, self.dim)

# ==========================================================
# 5. DIFFUSION-TS INTEGRATION SCRIPT
# ==========================================================
class Args_Example:
    def __init__(self) -> None:
        self.config_path = './Config/sines.yaml'
        self.save_dir = './sines_exp_idl'
        self.gpu = 0
        os.makedirs(self.save_dir, exist_ok=True)

def main():
    args = Args_Example()
    configs = load_yaml_config(args.config_path)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    dl_info = build_dataloader(configs, args)
    dataloader = dl_info['dataloader']
    dataset = dl_info['dataset']
    seq_length, feature_dim = dataset.window, dataset.var_num

    model = SequenceIDL(seq_len=seq_length, dim=feature_dim, n_flow=256, num_layers=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    
    epochs = 300
    
    # [NEW FIX 3] Added Linear Warmup to protect fragile initial weights
    warmup_epochs = 10
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs * len(dataloader)
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(epochs - warmup_epochs) * len(dataloader), eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs * len(dataloader)]
    )
    
    print(f"--- Training Joint Sequence IDL on {feature_dim}-D Sines Dataset ---")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            X = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            X = X.float() 
            
            optimizer.zero_grad()
            
            z, logdet = model(X)
            nll = 0.5 * model.p * math.log(2 * math.pi) + torch.mean(torch.sum(0.5 * z**2, -1) - logdet)
            
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += nll.item()
            
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

    print("\n--- Generating Fake Data (Simultaneous Pass) ---")
    chunk_size = 5000 
    num_samples = len(dataset)
    num_chunks = int(np.ceil(num_samples / chunk_size))
    
    fake_data_list = []
    for _ in tqdm(range(num_chunks)):
        size = min(chunk_size, num_samples - sum([len(c) for c in fake_data_list]))
        chunk = model.generate(batch_size=size, device=device)
        fake_data_list.append(chunk.cpu().numpy())
        
    fake_data = np.concatenate(fake_data_list, axis=0)
    
    if dataset.auto_norm:
        fake_data = unnormalize_to_zero_to_one(fake_data)
        
    np.save(os.path.join(args.save_dir, 'idl_fake_sine.npy'), fake_data)

if __name__ == "__main__":
    main()
