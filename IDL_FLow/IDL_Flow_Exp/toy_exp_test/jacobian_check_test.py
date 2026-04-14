import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
import math
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from implicit_function import ImplicitFunction, project_onto_Linf_ball


# 1. Custom Implicit Function using Tanh
class ImplicitFunctionTanh(ImplicitFunction):
    @staticmethod
    def phi(X: torch.Tensor) -> torch.Tensor:
        """ Tanh activation """
        return torch.tanh(X)

    @staticmethod
    def dphi(X: torch.Tensor) -> torch.Tensor:
        """ Derivative of Tanh: 1 - tanh(x)^2 """
        return 1.0 - torch.tanh(X)**2

# 2. Single IDL Flow Block with Tanh
class ImplicitFlowBlockTanh(nn.Module):
    def __init__(self, p=2, n=128, kappa=0.85):
        super().__init__()
        self.p = p
        self.n = n
        self.kappa = kappa

        # A, B, C, D parameterization
        self.A = nn.Parameter(torch.randn(n, n) / n)
        self.B = nn.Parameter(torch.randn(n, p) / math.sqrt(n))
        self.C = nn.Parameter(torch.randn(p, n) / math.sqrt(n))
        # Parameterize D to be strictly positive diagonal via exp() for invertibility
        self.d = nn.Parameter(torch.zeros(p))

    def forward(self, u):
        """ Maps Data (U) -> Latent (Z) """
        batch = u.shape[0]
        U = u.T

        A_proj = project_onto_Linf_ball(self.A, self.kappa)
        X0 = torch.zeros(self.n, batch, device=u.device)

        # Forward Implicit Pass
        X = ImplicitFunctionTanh.apply(A_proj, self.B, X0, U)

        D_diag = torch.exp(self.d)
        Y = self.C @ X + D_diag.unsqueeze(1) * U
        y = Y.T

        # Exact Jacobian Calculation
        Z_eq = A_proj @ X + self.B @ U
        Phi = ImplicitFunctionTanh.dphi(Z_eq)

        Phi_batch = Phi.T.unsqueeze(-1)
        A_batch = A_proj.unsqueeze(0).expand(batch, self.n, self.n)
        Phi_A = Phi_batch * A_batch

        B_batch = self.B.unsqueeze(0).expand(batch, self.n, self.p)
        Phi_B = Phi_batch * B_batch

        I = torch.eye(self.n, device=u.device).unsqueeze(0).expand(batch, self.n, self.n)
        V = torch.linalg.solve(I - Phi_A, Phi_B)

        C_batch = self.C.unsqueeze(0).expand(batch, self.p, self.n)
        D_batch = torch.diagflat(D_diag).unsqueeze(0).expand(batch, self.p, self.p)

        J = C_batch @ V + D_batch
        log_det_J = torch.log(torch.abs(torch.linalg.det(J)) + 1e-6)

        # Regularization penalty to ensure the inverse matrix A_tilde remains well-posed
        D_inv = torch.diagflat(1.0 / D_diag)
        A_tilde = A_proj - self.B @ D_inv @ self.C
        inv_penalty = F.relu(torch.linalg.matrix_norm(A_tilde, ord=float('inf')) - self.kappa).mean()

        return y, log_det_J, inv_penalty

    @torch.no_grad()
    def inverse(self, y):
        """ Maps Latent (Z) -> Data (U) """
        batch = y.shape[0]
        Y = y.T

        A_proj = project_onto_Linf_ball(self.A, self.kappa)
        D_diag = torch.exp(self.d)
        D_inv = torch.diagflat(1.0 / D_diag)

        A_tilde = A_proj - self.B @ D_inv @ self.C
        B_tilde = self.B @ D_inv

        X0 = torch.zeros(self.n, batch, device=y.device)
        X_inv = ImplicitFunctionTanh.apply(A_tilde, B_tilde, X0, Y)

        U_inv = D_inv @ Y - D_inv @ self.C @ X_inv
        return U_inv.T

# 3. Stacked IDL Flow
class StackedImplicitFlowTanh(nn.Module):
    def __init__(self, num_blocks=8, p=2, n=128, kappa=0.9):
        super().__init__()
        # Stack blocks to mimic the depth of 8 affine coupling layers
        self.blocks = nn.ModuleList([ImplicitFlowBlockTanh(p, n, kappa) for _ in range(num_blocks)])
        ImplicitFunctionTanh.set_parameters(mitr=500, grad_mitr=500, tol=1e-5, grad_tol=1e-5)

    def forward(self, u):
        total_log_det = 0
        total_inv_penalty = 0
        out = u
        for block in self.blocks:
            out, log_det, penalty = block(out)
            total_log_det += log_det
            total_inv_penalty += penalty
        return out, total_log_det, total_inv_penalty

    @torch.no_grad()
    def inverse(self, y):
        out = y
        for block in reversed(self.blocks):
            out = block.inverse(out)
        return out

def test_jacobian_exactness():
    print("Running Jacobian Exactness Test...")
    
    torch.set_default_dtype(torch.float64)
    
    p = 128
    n = 64
    batch_size = 1
    
    
    block = ImplicitFlowBlockTanh(p=p, n=n, kappa=0.9)
    block.eval() 

    ImplicitFunctionTanh.set_parameters(mitr=2000, grad_mitr=2000, tol=1e-12, grad_tol=1e-12)

    u = torch.randn(batch_size, p, requires_grad=True)
    
    def get_y(u_input):
        y, log_det_J, inv_penalty, J_analytical = block(u_input)
        return y
  
    J_autodiff = jacobian(get_y, u)
    J_autodiff = J_autodiff.squeeze()
    
    y, log_det_J, inv_penalty, J_analytical = block(u)
    J_analytical = J_analytical.squeeze()
    
    print("\n--- Autograd Jacobian (Iterative Backward Pass) ---")
    print(J_autodiff)
    
    print("\n--- Analytical Jacobian (Exact Matrix Inversion) ---")
    print(J_analytical)
    
    # 6. Assert exactness using torch.allclose
    # atol = absolute tolerance, rtol = relative tolerance
    is_exact = torch.allclose(J_autodiff, J_analytical, atol=1e-6, rtol=1e-5)
    
    assert is_exact, "FAILED: Analytical Jacobian does not match Autograd Jacobian!"
    print("\nSUCCESS! The mathematical implementation of the Jacobian is EXACT.")

if __name__ == "__main__":
    test_jacobian_exactness()