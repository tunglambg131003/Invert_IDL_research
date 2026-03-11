import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def create_masks(
    input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None
):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == "sequential":
        degrees += (
            [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [torch.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )

    elif input_order == "random":
        degrees += (
            [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [torch.randint(min_prev_degree, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.view(-1, x.shape[-1]).mean(0)
            # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)
            self.batch_var = x.view(-1, x.shape[-1]).var(0)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean.data * (1 - self.momentum)
            )
            self.running_var.mul_(self.momentum).add_(
                self.batch_var.data * (1 - self.momentum)
            )

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        #        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
        #            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer("mask", mask)

        # scale function
        s_net = [
            nn.Linear(
                input_size + (cond_label_size if cond_label_size is not None else 0),
                hidden_size,
            )
        ]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=-1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=-1)) * (
            1 - self.mask
        )

        # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)
        log_s = torch.tanh(s) * (1 - self.mask)
        u = x * torch.exp(log_s) + t
        # u = (x - t) * torch.exp(log_s)
        # u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)

        # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob
        # log_abs_det_jacobian = -(1 - self.mask) * s
        # log_abs_det_jacobian = -log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = log_s

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=-1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=-1)) * (
            1 - self.mask
        )

        log_s = torch.tanh(s) * (1 - self.mask)
        x = (u - t) * torch.exp(-log_s)
        # x = u * torch.exp(log_s) + t
        # x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        # log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du
        # log_abs_det_jacobian = log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = -log_s

        return x, log_abs_det_jacobian


class MaskedLinear(nn.Linear):
    """ MADE building block layer """

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer("mask", mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(
                torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size)
            )

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out


class MADE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="ReLU",
        input_order="sequential",
        input_degrees=None,
    ):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of MADEs
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )

        # setup activation
        if activation == "ReLU":
            activation_fn = nn.ReLU()
        elif activation == "Tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Check activation function.")

        # construct model
        self.net_input = MaskedLinear(
            input_size, hidden_size, masks[0], cond_label_size
        )
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1)),
        ]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        # D = u.shape[-1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
            x[..., i] = u[..., i] * torch.exp(loga[..., i]) + m[..., i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=-1)


class Flow(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.__scale = None
        self.net = None

        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def forward(self, x, cond):
        if self.scale is not None:
            x /= self.scale
        u, log_abs_det_jacobian = self.net(x, cond)
        return u, log_abs_det_jacobian

    def inverse(self, u, cond):
        x, log_abs_det_jacobian = self.net.inverse(u, cond)
        if self.scale is not None:
            x *= self.scale
            log_abs_det_jacobian += torch.log(torch.abs(self.scale))
        return x, log_abs_det_jacobian

    def log_prob(self, x, cond):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=-1)

    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1]
        else:
            shape = sample_shape

        u = self.base_dist.sample(shape)
        sample, _ = self.inverse(u, cond)
        return sample


class RealNVP(Flow):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        batch_norm=True,
    ):
        super().__init__(input_size)

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [
                LinearMaskedCoupling(
                    input_size, hidden_size, n_hidden, mask, cond_label_size
                )
            ]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)


class MAF(Flow):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="ReLU",
        input_order="sequential",
        batch_norm=True,
    ):
        super().__init__(input_size)

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [
                MADE(
                    input_size,
                    hidden_size,
                    n_hidden,
                    cond_label_size,
                    activation,
                    input_order,
                    self.input_degrees,
                )
            ]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

class EquilibriumSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z, A, mitr=1000, tol=3e-6):
       
        with torch.no_grad():
            x = torch.zeros_like(Z)
            
            for i in range(mitr):
                x_new = torch.tanh(F.linear(x, A) + Z)
                if torch.norm(x_new - x, p=float('inf')) < tol:
                    break
                x = x_new
                
        ctx.save_for_backward(A, x)
        ctx.mitr = mitr
        ctx.tol = tol
        return x

    @staticmethod
    def backward(ctx, grad_x):
        A, x = ctx.saved_tensors
        mitr = ctx.mitr
        tol = ctx.tol
        
        # Analytic gradients - also require no graph tracking!
        with torch.no_grad():
            dphi = 1.0 - x.pow(2) 
            V = torch.zeros_like(grad_x)
            dphi_grad_x = dphi * grad_x
            
            for i in range(mitr):
                V_new = dphi * F.linear(V, A.t()) + dphi_grad_x
                if torch.norm(V_new - V, p=float('inf')) < tol:
                    break
                V = V_new
                
            V_flat = V.reshape(-1, V.shape[-1])
            x_flat = x.reshape(-1, x.shape[-1])
            grad_A = torch.matmul(V_flat.t(), x_flat)
            grad_Z = V  
        
        return grad_Z, grad_A, None, None

class EquilibriumFlowLayer(nn.Module):
    def __init__(self, input_size, hidden_size, cond_label_size=None, 
                 idl_max_iter=1000, idl_tol=3e-6, idl_spectral_norm=0.95):
        super().__init__()
        self.input_size = input_size
        self.n_dim = hidden_size
        self.cond_size = cond_label_size
        
        self.idl_max_iter = idl_max_iter
        self.idl_tol = idl_tol
        self.idl_spectral_norm = idl_spectral_norm
        
        self.A = nn.Parameter(torch.randn(self.n_dim, self.n_dim) / self.n_dim)
        self.B = nn.Parameter(torch.randn(self.n_dim, input_size) / self.n_dim)
        self.C = nn.Parameter(torch.randn(input_size, self.n_dim) / self.n_dim)
        
        self.D = nn.Parameter(torch.eye(input_size) + torch.randn(input_size, input_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_size))
        
        if self.cond_size is not None:
            self.B_cond = nn.Parameter(torch.randn(self.n_dim, self.cond_size) / self.n_dim)
            self.D_cond = nn.Parameter(torch.randn(input_size, self.cond_size) / self.n_dim)
            
    def _project_A(self):
        """Exact L_inf projection to guarantee contraction mapping."""
        with torch.no_grad():
            v = self.idl_spectral_norm
            abs_A = self.A.abs()
            row_sums = abs_A.sum(dim=1)
            mask = row_sums > v
            
            if mask.any():
                for idx in torch.where(mask)[0]:
                    a_orig = self.A[idx]
                    a_sign = torch.sign(a_orig)
                    a_abs = torch.abs(a_orig)
                    a_sort, _ = torch.sort(a_abs)
                    
                    s = a_sort.sum() - v
                    l = float(len(a_sort))
                    for i in range(len(a_sort)):
                        if s / l > a_sort[i]:
                            s -= a_sort[i]
                            l -= 1
                        else:
                            break
                    alpha = s / l
                    self.A[idx] = a_sign * torch.clamp(a_abs - alpha, min=0)
                    
    def forward(self, inputs, cond=None):
        self._project_A()
        
        Z = F.linear(inputs, self.B)
        if self.cond_size is not None and cond is not None:
            Z = Z + F.linear(cond, self.B_cond)
            
        x = EquilibriumSolver.apply(Z, self.A, self.idl_max_iter, self.idl_tol)
        
        outputs = F.linear(x, self.C) + F.linear(inputs, self.D) + self.bias
        if self.cond_size is not None and cond is not None:
            outputs = outputs + F.linear(cond, self.D_cond)
            
        # -- FAST O(N^2) Truncated Log Det Jacobian --
        phi_prime = 1.0 - x.pow(2) 
        
        # A 3-step Neumann Series is incredibly fast and avoids OOMing the GPU.
        # It replaces the slow O(N^3) torch.linalg.solve with basic matrix multiplication
        M = phi_prime.unsqueeze(-1) * self.B
        for _ in range(3): 
            AM = torch.matmul(self.A, M)
            M = phi_prime.unsqueeze(-1) * self.B + phi_prime.unsqueeze(-1) * AM
            
        CM = torch.matmul(self.C, M)
        J = self.D + CM
        
        total_log_det = torch.slogdet(J)[1]
        
        log_abs_det_jacobian = (total_log_det / self.input_size).unsqueeze(-1).expand_as(inputs)
        
        return outputs, log_abs_det_jacobian

    def inverse(self, outputs, cond=None):
        D_inv = torch.inverse(self.D)
        
        z_shift = outputs - self.bias
        if self.cond_size is not None and cond is not None:
            z_shift = z_shift - F.linear(cond, self.D_cond)
            
        A_tilde = self.A - torch.matmul(self.B, torch.matmul(D_inv, self.C))
        B_tilde = torch.matmul(self.B, D_inv)
        
        Z_tilde = F.linear(z_shift, B_tilde)
        if self.cond_size is not None and cond is not None:
            Z_tilde = Z_tilde + F.linear(cond, self.B_cond)
            
        x = torch.zeros_like(Z_tilde)
        for _ in range(self.idl_max_iter):
            x_new = torch.tanh(F.linear(x, A_tilde) + Z_tilde)
            if torch.norm(x_new - x, p=float('inf')) < self.idl_tol:
                break
            x = x_new
            
        inputs = F.linear(z_shift - F.linear(x, self.C), D_inv)
        
        # Fast Inverse Log Det
        phi_prime = 1.0 - x.pow(2)
        M = phi_prime.unsqueeze(-1) * self.B
        for _ in range(3):
            AM = torch.matmul(self.A, M)
            M = phi_prime.unsqueeze(-1) * self.B + phi_prime.unsqueeze(-1) * AM
            
        CM = torch.matmul(self.C, M)
        J = self.D + CM
        
        total_log_det = -torch.slogdet(J)[1]
        
        # BROADCASTING FIX
        log_abs_det_jacobian = (total_log_det / self.input_size).unsqueeze(-1).expand_as(inputs)
        
        return inputs, log_abs_det_jacobian

class EquilibriumFlow(Flow):
    def __init__(self, input_size, hidden_size=100, n_blocks=3, cond_label_size=None,
                 idl_max_iter=1000, idl_tol=3e-6, idl_spectral_norm=0.95):
        super().__init__(input_size)
        modules = []
        for _ in range(n_blocks):
            modules.append(EquilibriumFlowLayer(
                input_size, hidden_size, cond_label_size,
                idl_max_iter, idl_tol, idl_spectral_norm
            ))
        self.net = FlowSequential(*modules)
