import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# ==========================================================
# 1. VECTORIZED BROYDEN'S SOLVER
# ==========================================================
def broyden_solver(f, x0, max_iter=50, tol=1e-6):
    x = x0.clone()
    n, batch = x.shape
    
    H = torch.eye(n, device=x.device).unsqueeze(0).expand(batch, n, n).clone()
    
    x_T = x.T 
    fx_T = f(x).T 
    
    step = fx_T.clone()
    x_prev_T = x_T.clone()
    fx_prev_T = fx_T.clone()
    x_T = x_T - step
    
    for _ in range(max_iter):
        fx_T = f(x_T.T).T
        
        if torch.max(torch.abs(fx_T)) < tol:
            break
            
        dx = (x_T - x_prev_T).unsqueeze(2)
        df = (fx_T - fx_prev_T).unsqueeze(2)
        
        H_df = torch.bmm(H, df)                   
        dx_T_H = torch.bmm(dx.transpose(1, 2), H) 
        denom = torch.bmm(dx_T_H, df)             
        
        denom_sign = torch.sign(denom)
        denom_sign[denom_sign == 0] = 1.0
        denom = denom + 1e-8 * denom_sign
        
        num = torch.bmm(dx - H_df, dx_T_H)        
        H = H + num / denom
        
        step = torch.bmm(H, fx_T.unsqueeze(2)).squeeze(2)
        
        x_prev_T = x_T.clone()
        fx_prev_T = fx_T.clone()
        x_T = x_T - step
        
    return x_T.T

# ==========================================================
# 2. BLACK-BOX BROYDEN ROOT FINDER (O(1) Memory)
# ==========================================================
class ImplicitRootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, U, b_x):
        batch = U.shape[1]
        n = A.shape[0]
        X_init = torch.zeros(n, batch, device=U.device)
        
        def f(X):
            return X - (1.0 / (2 * math.pi)) * torch.sin(2 * math.pi * (A @ X + B @ U + b_x))
            
        with torch.no_grad():
            X = broyden_solver(f, X_init, max_iter=50, tol=1e-6)
                
        ctx.save_for_backward(A, B, X, U, b_x)
        return X

    @staticmethod
    def backward(ctx, grad_X):
        A, B, X, U, b_x = ctx.saved_tensors
        
        with torch.no_grad():
            Z = A @ X + B @ U + b_x
            Phi = torch.cos(2 * math.pi * Z) 
            
            def f_adj(Y):
                return Y - grad_X - A.T @ (Phi * Y)
                
            Y_init = torch.zeros_like(grad_X)
            Y = broyden_solver(f_adj, Y_init, max_iter=50, tol=1e-5)
                
            Phi_Y = Phi * Y
            grad_A = Phi_Y @ X.T
            grad_B = Phi_Y @ U.T
            grad_U = B.T @ Phi_Y
            grad_b_x = Phi_Y.sum(dim=1, keepdim=True)
            
        return grad_A, grad_B, grad_U, grad_b_x

# ==========================================================
# 3. EXPRESSIVE PURE IDL BLOCK
# ==========================================================
class PureIDLFlowBlock(nn.Module):
    def __init__(self, p=2, n=128):
        super().__init__()
        self.p = p
        self.n = n
        
        self.A_layer = nn.utils.spectral_norm(nn.Linear(n, n, bias=False))
        self.B_layer = nn.utils.spectral_norm(nn.Linear(p, n, bias=False))
        self.C_layer = nn.utils.spectral_norm(nn.Linear(n, p, bias=False))
        
        self.b_x = nn.Parameter(torch.zeros(n, 1))
        self.b_y = nn.Parameter(torch.zeros(p, 1))
        
        self.register_buffer("D", torch.eye(p))
        
        self.p_A = nn.Parameter(torch.tensor(0.0))
        self.p_B = nn.Parameter(torch.tensor(0.0))
        self.p_C = nn.Parameter(torch.tensor(0.0))
        self.kappa = 0.98 
        
    def forward(self, u):
        batch = u.shape[0]
        U = u.T 
        
        _ = self.A_layer(torch.zeros(1, self.n, device=u.device))
        _ = self.B_layer(torch.zeros(1, self.p, device=u.device))
        _ = self.C_layer(torch.zeros(1, self.n, device=u.device))
        
        alpha = torch.sigmoid(self.p_A)
        beta = torch.exp(self.p_B)
        gamma = torch.sigmoid(self.p_C)
        
        A = self.A_layer.weight * (self.kappa * alpha)
        B = self.B_layer.weight * beta
        C = self.C_layer.weight * ((self.kappa * (1.0 - alpha) / beta) * gamma)
        
        X = ImplicitRootFinder.apply(A, B, U, self.b_x)
        
        Y = C @ X + self.D @ U + self.b_y
        y = Y.T
        
        Z_eq = A @ X + B @ U + self.b_x
        Phi = torch.cos(2 * math.pi * Z_eq)
        
        Phi_batch = Phi.T.unsqueeze(-1)
        A_batch = A.unsqueeze(0).expand(batch, self.n, self.n)
        Phi_A = Phi_batch * A_batch
        Phi_B = Phi_batch * B.unsqueeze(0).expand(batch, self.n, self.p)
        
        I = torch.eye(self.n, device=u.device).unsqueeze(0).expand(batch, self.n, self.n)
        V = torch.linalg.solve(I - Phi_A, Phi_B)
        
        J = C.unsqueeze(0).expand(batch, self.p, self.n) @ V + \
            self.D.unsqueeze(0).expand(batch, self.p, self.p)
            
        log_det_J = torch.log(torch.abs(torch.linalg.det(J)) + 1e-6)
        
        return y, log_det_J

    @torch.no_grad()
    def inverse(self, y):
        batch = y.shape[0]
        Y = y.T
        
        _ = self.A_layer(torch.zeros(1, self.n, device=y.device))
        _ = self.B_layer(torch.zeros(1, self.p, device=y.device))
        _ = self.C_layer(torch.zeros(1, self.n, device=y.device))
        
        alpha = torch.sigmoid(self.p_A)
        beta = torch.exp(self.p_B)
        gamma = torch.sigmoid(self.p_C)
        
        A = self.A_layer.weight * (self.kappa * alpha)
        B = self.B_layer.weight * beta
        C = self.C_layer.weight * ((self.kappa * (1.0 - alpha) / beta) * gamma)
        
        A_tilde = A - B @ C
        B_tilde = B 
        b_tilde = self.b_x - B @ self.b_y
        
        def f_inv(X):
            return X - (1.0 / (2 * math.pi)) * torch.sin(2 * math.pi * (A_tilde @ X + B_tilde @ Y + b_tilde))
            
        X_init = torch.zeros(self.n, batch, device=y.device)
        X = broyden_solver(f_inv, X_init, max_iter=50, tol=1e-5)
            
        U_inv = Y - C @ X - self.b_y
        return U_inv.T

# ==========================================================
# 4. STACKED FLOW
# ==========================================================
class StackedExpressiveIDL(nn.Module):
    def __init__(self, num_blocks=8, p=2, n=128):
        super().__init__()
        self.blocks = nn.ModuleList([PureIDLFlowBlock(p, n) for _ in range(num_blocks)])
        
    def forward(self, u):
        total_log_det = 0
        out = u
        for block in self.blocks:
            out, log_det = block(out)
            total_log_det += log_det
        return out, total_log_det
        
    @torch.no_grad()
    def inverse(self, y):
        out = y
        for block in reversed(self.blocks):
            out = block.inverse(out)
        return out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    idl_flow = StackedExpressiveIDL(num_blocks=8, p=2, n=128).to(device)
    optimizer = torch.optim.Adam(idl_flow.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15000, eta_min=1e-5)
    
    num_steps = 15000
    
    print("Training Expressive Pure IDL Flow with Broyden's Method (CIRCLES)...")
    for idx_step in range(num_steps):
     
        X, _ = datasets.make_circles(n_samples=512, noise=0.05, factor=0.5)
        X = torch.Tensor(X).to(device)
    
        z, logdet = idl_flow(X)
        loss = torch.log(z.new_tensor([2 * math.pi])) + torch.mean(torch.sum(0.5 * z**2, -1) - logdet)
    
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(idl_flow.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
        if (idx_step + 1) % 500 == 0:
            print(f"idx_steps: {idx_step + 1}, NLL loss: {loss.item():.5f}")
    
    # ------------------ Plotting ------------------
    z_samples = torch.randn(2000, 2).to(device)
    x_generated = idl_flow.inverse(z_samples)
    
    z_samples_np = z_samples.cpu().detach().numpy()
    x_generated_np = x_generated.cpu().detach().numpy()
    
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(z_samples_np[:, 0], z_samples_np[:, 1], alpha=0.5, s=10, c='gray')
    plt.title("Latent Space N(0, I)")
    plt.subplot(1, 2, 2)
    plt.scatter(x_generated_np[:, 0], x_generated_np[:, 1], alpha=0.5, s=10, c='red')
    plt.title("Generated Circles (Broyden Pure IDL)")
    plt.savefig('idl_spectral_circles.png')
    print("Saved generation plot.")
    plt.show()

if __name__ == "__main__":
    main()
