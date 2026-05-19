import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm

# ---- Diffusion-TS Imports ----
from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

torch.backends.cuda.preferred_linalg_library("cusolver")

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
# 2. BLACK-BOX ROOT FINDER (SOFTPLUS BETA-LIPSWISH)
# ==========================================================
class ImplicitRootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, U, b_x, beta):
        batch = U.shape[1]
        n = A.shape[0]
        X_init = torch.zeros(n, batch, device=U.device)
        
        # Calculate strictly positive effective beta
        beta_eff = F.softplus(beta)
        
        def f(X):
            Z_eq = A @ X + B @ U + b_x
            # Picard Iteration requires the Next State (Contraction Mapping)
            return (Z_eq * torch.sigmoid(beta_eff * Z_eq)) / 1.1
            
        with torch.no_grad():
            X = fixed_point_solver(f, X_init, max_iter=300, tol=1e-9) 
                
        ctx.save_for_backward(A, B, X, U, b_x, beta, beta_eff)
        return X

    @staticmethod
    def backward(ctx, grad_X):
        A, B, X, U, b_x, beta, beta_eff = ctx.saved_tensors
        with torch.no_grad():
            Z_eq = A @ X + B @ U + b_x
            S = torch.sigmoid(beta_eff * Z_eq)
            
            # Exact Derivative w.r.t Z using effective beta
            Phi = (S + beta_eff * Z_eq * S * (1.0 - S)) / 1.1 
            
            def f_adj(Y):
                # The adjoint system is also a contraction mapping
                return grad_X + A.T @ (Phi * Y)
                
            Y_init = torch.zeros_like(grad_X)
            Y = fixed_point_solver(f_adj, Y_init, max_iter=300, tol=1e-10)
                
            Phi_Y = Phi * Y
            grad_A, grad_B = Phi_Y @ X.T, Phi_Y @ U.T
            grad_U, grad_b_x = B.T @ Phi_Y, Phi_Y.sum(dim=1, keepdim=True)
            
            # Exact Derivative w.r.t raw Beta (Chain rule through Softplus)
            grad_beta_eff = (Z_eq**2 * S * (1.0 - S)) / 1.1
            grad_beta_raw = grad_beta_eff * torch.sigmoid(beta)
            grad_beta = (grad_beta_raw * Y).sum(dim=1, keepdim=True)
            
        return grad_A, grad_B, grad_U, grad_b_x, grad_beta

# ==========================================================
# 3. PURE IDL BLOCK (Sequence-Level)
# ==========================================================
class PureIDLFlowBlock(nn.Module):
    def __init__(self, p, n=128):
        super().__init__()
        self.p, self.n = p, n
        
        # Added n_power_iterations=20 to ensure strict Lipschitz bound
        self.A_layer = nn.utils.spectral_norm(nn.Linear(n, n, bias=False), n_power_iterations=20)
        self.B_layer = nn.utils.spectral_norm(nn.Linear(p, n, bias=False), n_power_iterations=20)
        self.C_layer = nn.utils.spectral_norm(nn.Linear(n, p, bias=False), n_power_iterations=20)
        
        self.b_x = nn.Parameter(torch.zeros(n, 1))
        self.b_y = nn.Parameter(torch.zeros(p, 1))
        
        # Initialize learnable beta parameter at 0.5
        self.beta = nn.Parameter(torch.full((n, 1), 0.5))
        
        self.register_buffer("D", torch.eye(p))
        
        self.p_A = nn.Parameter(torch.tensor(0.0))
        self.p_B = nn.Parameter(torch.tensor(0.0))
        self.p_C = nn.Parameter(torch.tensor(0.0))
        self.kappa = 0.99
        
    def forward(self, u):
        batch = u.shape[0]
        U = u.T 
        
        _ = self.A_layer(torch.zeros(1, self.n, device=u.device))
        _ = self.B_layer(torch.zeros(1, self.p, device=u.device))
        _ = self.C_layer(torch.zeros(1, self.n, device=u.device))
        
        # Renamed spectral norm beta to beta_weight to avoid shadowing self.beta
        alpha, beta_weight, gamma = torch.sigmoid(self.p_A), torch.exp(self.p_B), torch.sigmoid(self.p_C)
        A = self.A_layer.weight * (self.kappa * alpha)
        B = self.B_layer.weight * beta_weight
        C = self.C_layer.weight * ((self.kappa * (1.0 - alpha) / beta_weight) * gamma)
        
        # Pass the raw beta into the implicit solver
        X = ImplicitRootFinder.apply(A, B, U, self.b_x, self.beta)
        
        Y = C @ X + self.D @ U + self.b_y
        y = Y.T 
        
        # Log-Det calculation with effective beta
        beta_eff = F.softplus(self.beta)
        Z_eq = A @ X + B @ U + self.b_x
        S = torch.sigmoid(beta_eff * Z_eq)
        Phi = (S + beta_eff * Z_eq * S * (1.0 - S)) / 1.1
        Phi_batch = Phi.T.unsqueeze(-1)
        
        I = torch.eye(self.n, device=u.device).unsqueeze(0).expand(batch, self.n, self.n)
        A_eff = I - Phi_batch * A.unsqueeze(0)
        A_eff = A_eff + torch.eye(self.n, device=u.device).unsqueeze(0) * 1e-5
        
        V = torch.linalg.solve(A_eff, Phi_batch * B.unsqueeze(0))
        
        J = C.unsqueeze(0).expand(batch, self.p, self.n) @ V + self.D.unsqueeze(0).expand(batch, self.p, self.p)
        
        # Replaced torch.det with slogdet to prevent FP32 underflow
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
        
        alpha, beta_weight, gamma = torch.sigmoid(self.p_A), torch.exp(self.p_B), torch.sigmoid(self.p_C)
        A = self.A_layer.weight * (self.kappa * alpha)
        B = self.B_layer.weight * beta_weight
        C = self.C_layer.weight * ((self.kappa * (1.0 - alpha) / beta_weight) * gamma)
        
        A_tilde, B_tilde = A - B @ C, B 
        b_tilde = self.b_x - B @ self.b_y 
        
        beta_eff = F.softplus(self.beta)
        
        def f_inv(X):
            Z_eq = A_tilde @ X + B_tilde @ Y + b_tilde
            # Picard Iteration requires the Next State
            return (Z_eq * torch.sigmoid(beta_eff * Z_eq)) / 1.1
            
        X = fixed_point_solver(f_inv, torch.zeros(self.n, batch, device=y.device), max_iter=400, tol=1e-10)
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

class EarlyStopping:
    def __init__(self, patience=25, min_delta=1e-4, save_path='best_idl_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, current_loss, model, optimizer, scheduler, epoch):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            
            # Save FULL Checkpoint Dictionary
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': self.best_loss
            }
            torch.save(checkpoint, self.save_path)
            # print(f"  -> Model saved at epoch {epoch+1}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==========================================================
# 6. DIFFUSION-TS INTEGRATION SCRIPT
# ==========================================================
class Args_Example:
    def __init__(self) -> None:
        self.config_path = './Config/etth.yaml'
        self.save_dir = './ETTh_exp_idl'
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

    model = SequenceIDL(seq_len=seq_length, dim=feature_dim, n_flow=512, num_layers=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-6)
    
    epochs = 350
    
    # Added Linear Warmup to protect fragile initial weights
    warmup_epochs = 10
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs * len(dataloader)
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(epochs - warmup_epochs) * len(dataloader), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs * len(dataloader)]
    )

    model_save_path = os.path.join(args.save_dir, 'best_idl_model.pth')
    early_stopping = EarlyStopping(patience=25, min_delta=1e-5, save_path=model_save_path)

    start_epoch = 0
    if os.path.exists(model_save_path):
        try:
            checkpoint = torch.load(model_save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            early_stopping.best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"\n✅ Found checkpoint! Successfully resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"\n⚠️ Could not resume perfectly (likely an older weight file). Starting fresh or from epoch 0. Error: {e}")
            model.load_state_dict(torch.load(model_save_path, weights_only=True)) # Fallback if dict missing
    
    print(f"--- Training Joint Sequence IDL on {feature_dim}-D ETTh Dataset ---")
    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            X = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            X = X.float() 

            if model.training:
                X = X + torch.randn_like(X) * 0.0001
            
            optimizer.zero_grad()
            
            z, logdet = model(X)
            nll = 0.5 * model.p * math.log(2 * math.pi) + torch.mean(torch.sum(0.5 * z**2, -1) - logdet)
            
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += nll.item()
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f}")

        # ==========================================================
        # SAVE CHECKPOINTS AT FIXED EPOCHS
        # ==========================================================
        save_epochs = [50, 100, 150, 200, 250, 300, 350]

        if (epoch + 1) in save_epochs:
            checkpoint_path = os.path.join(
            args.save_dir,
            f'idl_checkpoint_epoch_{epoch+1}.pth'
            )

            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': early_stopping.best_loss
        }

            torch.save(checkpoint, checkpoint_path)

            print(f"✅ Saved checkpoint at epoch {epoch+1}")
        
        # ==========================================================
        # EARLY STOPPING
        # ==========================================================

        # Pass EVERYTHING to early stopping so it can build the checkpoint

        early_stopping(avg_epoch_loss, model, optimizer, scheduler, epoch)
        
        if early_stopping.early_stop:
            print(f"\n--- Early stopping triggered at Epoch {epoch+1}! ---")
            break

    # Load best model for generation
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # fallback

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
        
    np.save(os.path.join(args.save_dir, 'idl_fake_ETTh.npy'), fake_data)

if __name__ == "__main__":
    main()