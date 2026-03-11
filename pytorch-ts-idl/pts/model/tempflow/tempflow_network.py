# from typing import List, Optional, Tuple, Union

# import torch
# import torch.nn as nn

# from gluonts.core.component import validated

# from pts.model import weighted_average
# from pts.modules import RealNVP, MAF, FlowOutput, MeanScaler, NOPScaler


# class TempFlowTrainingNetwork(nn.Module):
#     @validated()
#     def __init__(
#         self,
#         input_size: int,
#         num_layers: int,
#         num_cells: int,
#         cell_type: str,
#         history_length: int,
#         context_length: int,
#         prediction_length: int,
#         dropout_rate: float,
#         lags_seq: List[int],
#         target_dim: int,
#         conditioning_length: int,
#         flow_type: str,
#         n_blocks: int,
#         hidden_size: int,
#         n_hidden: int,
#         dequantize: bool,
#         cardinality: List[int] = [1],
#         embedding_dimension: int = 1,
#         scaling: bool = True,
#         **kwargs,
#     ) -> None:
#         super().__init__(**kwargs)
#         self.target_dim = target_dim
#         self.prediction_length = prediction_length
#         self.context_length = context_length
#         self.history_length = history_length
#         self.scaling = scaling

#         assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
#         lags_seq.sort()
#         self.lags_seq = lags_seq

#         self.cell_type = cell_type
#         rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
#         self.rnn = rnn_cls(
#             input_size=input_size,
#             hidden_size=num_cells,
#             num_layers=num_layers,
#             dropout=dropout_rate,
#             batch_first=True,
#         )

#         flow_cls = {
#             "RealNVP": RealNVP,
#             "MAF": MAF,
#         }[flow_type]
#         self.flow = flow_cls(
#             input_size=target_dim,
#             n_blocks=n_blocks,
#             n_hidden=n_hidden,
#             hidden_size=hidden_size,
#             cond_label_size=conditioning_length,
#         )
#         self.dequantize = dequantize

#         self.distr_output = FlowOutput(
#             self.flow, input_size=target_dim, cond_size=conditioning_length
#         )

#         self.proj_dist_args = self.distr_output.get_args_proj(num_cells)

#         self.embed_dim = 1
#         self.embed = nn.Embedding(
#             num_embeddings=self.target_dim, embedding_dim=self.embed_dim
#         )

#         if self.scaling:
#             self.scaler = MeanScaler(keepdim=True)
#         else:
#             self.scaler = NOPScaler(keepdim=True)

#     @staticmethod
#     def get_lagged_subsequences(
#         sequence: torch.Tensor,
#         sequence_length: int,
#         indices: List[int],
#         subsequences_length: int = 1,
#     ) -> torch.Tensor:
#         """
#         Returns lagged subsequences of a given sequence.
#         Parameters
#         ----------
#         sequence
#             the sequence from which lagged subsequences should be extracted.
#             Shape: (N, T, C).
#         sequence_length
#             length of sequence in the T (time) dimension (axis = 1).
#         indices
#             list of lag indices to be used.
#         subsequences_length
#             length of the subsequences to be extracted.
#         Returns
#         --------
#         lagged : Tensor
#             a tensor of shape (N, S, C, I),
#             where S = subsequences_length and I = len(indices),
#             containing lagged subsequences.
#             Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
#         """
#         # we must have: history_length + begin_index >= 0
#         # that is: history_length - lag_index - sequence_length >= 0
#         # hence the following assert
#         assert max(indices) + subsequences_length <= sequence_length, (
#             f"lags cannot go further than history length, found lag "
#             f"{max(indices)} while history length is only {sequence_length}"
#         )
#         assert all(lag_index >= 0 for lag_index in indices)

#         lagged_values = []
#         for lag_index in indices:
#             begin_index = -lag_index - subsequences_length
#             end_index = -lag_index if lag_index > 0 else None
#             lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
#         return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

#     def unroll(
#         self,
#         lags: torch.Tensor,
#         scale: torch.Tensor,
#         time_feat: torch.Tensor,
#         target_dimension_indicator: torch.Tensor,
#         unroll_length: int,
#         begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
#     ) -> Tuple[
#         torch.Tensor,
#         Union[List[torch.Tensor], torch.Tensor],
#         torch.Tensor,
#         torch.Tensor,
#     ]:

#         # (batch_size, sub_seq_len, target_dim, num_lags)
#         lags_scaled = lags / scale.unsqueeze(-1)

#         # assert_shape(
#         #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
#         # )

#         input_lags = lags_scaled.reshape(
#             (-1, unroll_length, len(self.lags_seq) * self.target_dim)
#         )

#         # (batch_size, target_dim, embed_dim)
#         index_embeddings = self.embed(target_dimension_indicator)
#         # assert_shape(index_embeddings, (-1, self.target_dim, self.embed_dim))

#         # (batch_size, seq_len, target_dim * embed_dim)
#         repeated_index_embeddings = (
#             index_embeddings.unsqueeze(1)
#             .expand(-1, unroll_length, -1, -1)
#             .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
#         )

#         # (batch_size, sub_seq_len, input_dim)
#         inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1)

#         # unroll encoder
#         outputs, state = self.rnn(inputs, begin_state)

#         # assert_shape(outputs, (-1, unroll_length, self.num_cells))
#         # for s in state:
#         #     assert_shape(s, (-1, self.num_cells))

#         # assert_shape(
#         #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
#         # )

#         return outputs, state, lags_scaled, inputs

#     def unroll_encoder(
#         self,
#         past_time_feat: torch.Tensor,
#         past_target_cdf: torch.Tensor,
#         past_observed_values: torch.Tensor,
#         past_is_pad: torch.Tensor,
#         future_time_feat: Optional[torch.Tensor],
#         future_target_cdf: Optional[torch.Tensor],
#         target_dimension_indicator: torch.Tensor,
#     ) -> Tuple[
#         torch.Tensor,
#         Union[List[torch.Tensor], torch.Tensor],
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#     ]:
#         """
#         Unrolls the RNN encoder over past and, if present, future data.
#         Returns outputs and state of the encoder, plus the scale of
#         past_target_cdf and a vector of static features that was constructed
#         and fed as input to the encoder. All tensor arguments should have NTC
#         layout.

#         Parameters
#         ----------
#         past_time_feat
#             Past time features (batch_size, history_length, num_features)
#         past_target_cdf
#             Past marginal CDF transformed target values (batch_size,
#             history_length, target_dim)
#         past_observed_values
#             Indicator whether or not the values were observed (batch_size,
#             history_length, target_dim)
#         past_is_pad
#             Indicator whether the past target values have been padded
#             (batch_size, history_length)
#         future_time_feat
#             Future time features (batch_size, prediction_length, num_features)
#         future_target_cdf
#             Future marginal CDF transformed target values (batch_size,
#             prediction_length, target_dim)
#         target_dimension_indicator
#             Dimensionality of the time series (batch_size, target_dim)

#         Returns
#         -------
#         outputs
#             RNN outputs (batch_size, seq_len, num_cells)
#         states
#             RNN states. Nested list with (batch_size, num_cells) tensors with
#         dimensions target_dim x num_layers x (batch_size, num_cells)
#         scale
#             Mean scales for the time series (batch_size, 1, target_dim)
#         lags_scaled
#             Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
#         inputs
#             inputs to the RNN

#         """

#         past_observed_values = torch.min(
#             past_observed_values, 1 - past_is_pad.unsqueeze(-1)
#         )

#         if future_time_feat is None or future_target_cdf is None:
#             time_feat = past_time_feat[:, -self.context_length :, ...]
#             sequence = past_target_cdf
#             sequence_length = self.history_length
#             subsequences_length = self.context_length
#         else:
#             time_feat = torch.cat(
#                 (past_time_feat[:, -self.context_length :, ...], future_time_feat),
#                 dim=1,
#             )
#             sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
#             sequence_length = self.history_length + self.prediction_length
#             subsequences_length = self.context_length + self.prediction_length

#         # (batch_size, sub_seq_len, target_dim, num_lags)
#         lags = self.get_lagged_subsequences(
#             sequence=sequence,
#             sequence_length=sequence_length,
#             indices=self.lags_seq,
#             subsequences_length=subsequences_length,
#         )

#         # scale is computed on the context length last units of the past target
#         # scale shape is (batch_size, 1, target_dim)
#         _, scale = self.scaler(
#             past_target_cdf[:, -self.context_length :, ...],
#             past_observed_values[:, -self.context_length :, ...],
#         )

#         outputs, states, lags_scaled, inputs = self.unroll(
#             lags=lags,
#             scale=scale,
#             time_feat=time_feat,
#             target_dimension_indicator=target_dimension_indicator,
#             unroll_length=subsequences_length,
#             begin_state=None,
#         )

#         return outputs, states, scale, lags_scaled, inputs

#     def distr_args(self, rnn_outputs: torch.Tensor):
#         """
#         Returns the distribution of DeepVAR with respect to the RNN outputs.

#         Parameters
#         ----------
#         rnn_outputs
#             Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
#         scale
#             Mean scale for each time series (batch_size, 1, target_dim)

#         Returns
#         -------
#         distr
#             Distribution instance
#         distr_args
#             Distribution arguments
#         """
#         (distr_args,) = self.proj_dist_args(rnn_outputs)

#         # # compute likelihood of target given the predicted parameters
#         # distr = self.distr_output.distribution(distr_args, scale=scale)

#         # return distr, distr_args
#         return distr_args

#     def forward(
#         self,
#         target_dimension_indicator: torch.Tensor,
#         past_time_feat: torch.Tensor,
#         past_target_cdf: torch.Tensor,
#         past_observed_values: torch.Tensor,
#         past_is_pad: torch.Tensor,
#         future_time_feat: torch.Tensor,
#         future_target_cdf: torch.Tensor,
#         future_observed_values: torch.Tensor,
#     ) -> Tuple[torch.Tensor, ...]:
#         """
#         Computes the loss for training DeepVAR, all inputs tensors representing
#         time series have NTC layout.

#         Parameters
#         ----------
#         target_dimension_indicator
#             Indices of the target dimension (batch_size, target_dim)
#         past_time_feat
#             Dynamic features of past time series (batch_size, history_length,
#             num_features)
#         past_target_cdf
#             Past marginal CDF transformed target values (batch_size,
#             history_length, target_dim)
#         past_observed_values
#             Indicator whether or not the values were observed (batch_size,
#             history_length, target_dim)
#         past_is_pad
#             Indicator whether the past target values have been padded
#             (batch_size, history_length)
#         future_time_feat
#             Future time features (batch_size, prediction_length, num_features)
#         future_target_cdf
#             Future marginal CDF transformed target values (batch_size,
#             prediction_length, target_dim)
#         future_observed_values
#             Indicator whether or not the future values were observed
#             (batch_size, prediction_length, target_dim)

#         Returns
#         -------
#         distr
#             Loss with shape (batch_size, 1)
#         likelihoods
#             Likelihoods for each time step
#             (batch_size, context + prediction_length, 1)
#         distr_args
#             Distribution arguments (context + prediction_length,
#             number_of_arguments)
#         """

#         seq_len = self.context_length + self.prediction_length

#         # unroll the decoder in "training mode", i.e. by providing future data
#         # as well
#         rnn_outputs, _, scale, _, _ = self.unroll_encoder(
#             past_time_feat=past_time_feat,
#             past_target_cdf=past_target_cdf,
#             past_observed_values=past_observed_values,
#             past_is_pad=past_is_pad,
#             future_time_feat=future_time_feat,
#             future_target_cdf=future_target_cdf,
#             target_dimension_indicator=target_dimension_indicator,
#         )

#         # put together target sequence
#         # (batch_size, seq_len, target_dim)
#         target = torch.cat(
#             (past_target_cdf[:, -self.context_length :, ...], future_target_cdf),
#             dim=1,
#         )

#         # assert_shape(target, (-1, seq_len, self.target_dim))

#         distr_args = self.distr_args(rnn_outputs=rnn_outputs)
#         if self.scaling:
#             self.flow.scale = scale

#         # we sum the last axis to have the same shape for all likelihoods
#         # (batch_size, subseq_length, 1)
#         if self.dequantize:
#             target += torch.rand_like(target)
#         likelihoods = -self.flow.log_prob(target, distr_args).unsqueeze(-1)

#         # assert_shape(likelihoods, (-1, seq_len, 1))

#         past_observed_values = torch.min(
#             past_observed_values, 1 - past_is_pad.unsqueeze(-1)
#         )

#         # (batch_size, subseq_length, target_dim)
#         observed_values = torch.cat(
#             (
#                 past_observed_values[:, -self.context_length :, ...],
#                 future_observed_values,
#             ),
#             dim=1,
#         )

#         # mask the loss at one time step if one or more observations is missing
#         # in the target dimensions (batch_size, subseq_length, 1)
#         loss_weights, _ = observed_values.min(dim=-1, keepdim=True)

#         # assert_shape(loss_weights, (-1, seq_len, 1))

#         loss = weighted_average(likelihoods, weights=loss_weights, dim=1)

#         # assert_shape(loss, (-1, -1, 1))

#         # self.distribution = distr

#         return (loss.mean(), likelihoods, distr_args)


# class TempFlowPredictionNetwork(TempFlowTrainingNetwork):
#     def __init__(self, num_parallel_samples: int, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.num_parallel_samples = num_parallel_samples

#         # for decoding the lags are shifted by one,
#         # at the first time-step of the decoder a lag of one corresponds to
#         # the last target value
#         self.shifted_lags = [l - 1 for l in self.lags_seq]

#     def sampling_decoder(
#         self,
#         past_target_cdf: torch.Tensor,
#         target_dimension_indicator: torch.Tensor,
#         time_feat: torch.Tensor,
#         scale: torch.Tensor,
#         begin_states: Union[List[torch.Tensor], torch.Tensor],
#     ) -> torch.Tensor:
#         """
#         Computes sample paths by unrolling the RNN starting with a initial
#         input and state.

#         Parameters
#         ----------
#         past_target_cdf
#             Past marginal CDF transformed target values (batch_size,
#             history_length, target_dim)
#         target_dimension_indicator
#             Indices of the target dimension (batch_size, target_dim)
#         time_feat
#             Dynamic features of future time series (batch_size, history_length,
#             num_features)
#         scale
#             Mean scale for each time series (batch_size, 1, target_dim)
#         begin_states
#             List of initial states for the RNN layers (batch_size, num_cells)
#         Returns
#         --------
#         sample_paths : Tensor
#             A tensor containing sampled paths. Shape: (1, num_sample_paths,
#             prediction_length, target_dim).
#         """

#         def repeat(tensor, dim=0):
#             return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

#         # blows-up the dimension of each tensor to
#         # batch_size * self.num_sample_paths for increasing parallelism
#         repeated_past_target_cdf = repeat(past_target_cdf)
#         repeated_time_feat = repeat(time_feat)
#         repeated_scale = repeat(scale)
#         if self.scaling:
#             self.flow.scale = repeated_scale
#         repeated_target_dimension_indicator = repeat(target_dimension_indicator)

#         if self.cell_type == "LSTM":
#             repeated_states = [repeat(s, dim=1) for s in begin_states]
#         else:
#             repeated_states = repeat(begin_states, dim=1)

#         future_samples = []

#         # for each future time-units we draw new samples for this time-unit
#         # and update the state
#         for k in range(self.prediction_length):
#             lags = self.get_lagged_subsequences(
#                 sequence=repeated_past_target_cdf,
#                 sequence_length=self.history_length + k,
#                 indices=self.shifted_lags,
#                 subsequences_length=1,
#             )

#             rnn_outputs, repeated_states, _, _ = self.unroll(
#                 begin_state=repeated_states,
#                 lags=lags,
#                 scale=repeated_scale,
#                 time_feat=repeated_time_feat[:, k : k + 1, ...],
#                 target_dimension_indicator=repeated_target_dimension_indicator,
#                 unroll_length=1,
#             )

#             distr_args = self.distr_args(rnn_outputs=rnn_outputs)

#             # (batch_size, 1, target_dim)
#             new_samples = self.flow.sample(cond=distr_args)

#             # (batch_size, seq_len, target_dim)
#             future_samples.append(new_samples)
#             repeated_past_target_cdf = torch.cat(
#                 (repeated_past_target_cdf, new_samples), dim=1
#             )

#         # (batch_size * num_samples, prediction_length, target_dim)
#         samples = torch.cat(future_samples, dim=1)

#         # (batch_size, num_samples, prediction_length, target_dim)
#         return samples.reshape(
#             (
#                 -1,
#                 self.num_parallel_samples,
#                 self.prediction_length,
#                 self.target_dim,
#             )
#         )

#     def forward(
#         self,
#         target_dimension_indicator: torch.Tensor,
#         past_time_feat: torch.Tensor,
#         past_target_cdf: torch.Tensor,
#         past_observed_values: torch.Tensor,
#         past_is_pad: torch.Tensor,
#         future_time_feat: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Predicts samples given the trained DeepVAR model.
#         All tensors should have NTC layout.
#         Parameters
#         ----------
#         target_dimension_indicator
#             Indices of the target dimension (batch_size, target_dim)
#         past_time_feat
#             Dynamic features of past time series (batch_size, history_length,
#             num_features)
#         past_target_cdf
#             Past marginal CDF transformed target values (batch_size,
#             history_length, target_dim)
#         past_observed_values
#             Indicator whether or not the values were observed (batch_size,
#             history_length, target_dim)
#         past_is_pad
#             Indicator whether the past target values have been padded
#             (batch_size, history_length)
#         future_time_feat
#             Future time features (batch_size, prediction_length, num_features)

#         Returns
#         -------
#         sample_paths : Tensor
#             A tensor containing sampled paths (1, num_sample_paths,
#             prediction_length, target_dim).

#         """

#         # mark padded data as unobserved
#         # (batch_size, target_dim, seq_len)
#         past_observed_values = torch.min(
#             past_observed_values, 1 - past_is_pad.unsqueeze(-1)
#         )

#         # unroll the decoder in "prediction mode", i.e. with past data only
#         _, begin_states, scale, _, _ = self.unroll_encoder(
#             past_time_feat=past_time_feat,
#             past_target_cdf=past_target_cdf,
#             past_observed_values=past_observed_values,
#             past_is_pad=past_is_pad,
#             future_time_feat=None,
#             future_target_cdf=None,
#             target_dimension_indicator=target_dimension_indicator,
#         )

#         return self.sampling_decoder(
#             past_target_cdf=past_target_cdf,
#             target_dimension_indicator=target_dimension_indicator,
#             time_feat=future_time_feat,
#             scale=scale,
#             begin_states=begin_states,
#         )

# from typing import List, Optional, Tuple, Union
# import warnings
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Function

# from gluonts.core.component import validated

# from pts.model import weighted_average
# from pts.modules import RealNVP, MAF, FlowOutput, MeanScaler, NOPScaler


# # -------------------------------------------------------------------------
# # Implicit Deep Learning Components
# # -------------------------------------------------------------------------

# def transpose(x):
#     return x.transpose(-2, -1)

# class ImplicitFunctionWarning(RuntimeWarning):
#     pass

# class ImplicitFunction(Function):
#     @classmethod
#     def forward(cls, ctx, A, B, X0, U, max_iter, tol):
#         ctx.max_iter = max_iter
#         ctx.tol = tol
#         with torch.no_grad():
#             X, err, status = cls.inn_pred(A, B @ U, X0, max_iter, tol)
#         ctx.save_for_backward(A, B, X, U)
#         if status != "converged":
#             # Warn only if error is significantly above tolerance to avoid log spam
#             if err > tol * 10: 
#                 warnings.warn(f"Picard iterations did not converge: err={err.item():.4e}, status={status}", ImplicitFunctionWarning)
#         return X

#     @classmethod
#     def backward(cls, ctx, *grad_outputs):
#         A, B, X, U = ctx.saved_tensors
#         grad_output = grad_outputs[0]
#         max_iter = ctx.max_iter
#         tol = ctx.tol

#         DPhi = cls.dphi(A @ X + B @ U)
#         V, err, status = cls.inn_pred_grad(A.T, DPhi * grad_output, DPhi, max_iter, tol)
        
#         if status != "converged":
#              if err > tol * 10:
#                 warnings.warn(f"Gradient iterations did not converge: err={err.item():.4e}, status={status}", ImplicitFunctionWarning)
        
#         grad_A = V @ X.T
#         grad_B = V @ U.T
#         grad_U = B.T @ V
#         # Return grads for A, B, X0, U, and None for scalar args
#         return grad_A, grad_B, torch.zeros_like(X), grad_U, None, None

#     @staticmethod
#     def phi(X):
#         return torch.tanh(X)

#     @staticmethod
#     def dphi(X):
    
#         return 1.0 - torch.tanh(X).pow(2)

#     @classmethod
#     def inn_pred(cls, A, Z, X, mitr, tol):
#         err = 0
#         status = 'max itrs reached'
#         for i in range(mitr):
#             X_new = cls.phi(A @ X + Z)
#             err = torch.norm(X_new - X, float('inf'))
#             if err < tol:
#                 status = 'converged'
#                 break
#             X = X_new
#         return X, err, status

#     @staticmethod
#     def inn_pred_grad(AT, Z, DPhi, mitr, tol):
#         X = torch.zeros_like(Z)
#         err = 0
#         status = 'max itrs reached'
#         for i in range(mitr):
#             X_new = DPhi * (AT @ X) + Z
#             err = torch.norm(X_new - X, float('inf'))
#             if err < tol:
#                 status = 'converged'
#                 break
#             X = X_new
#         return X, err, status


# class ImplicitFunctionInf(ImplicitFunction):
#     @classmethod
#     def forward(cls, ctx, A, B, X0, U, max_iter, tol, spectral_norm):
#         v = spectral_norm
#         A_np = A.clone().detach().cpu().numpy()
#         x = np.abs(A_np).sum(axis=-1)
#         mask = x > v
#         if mask.any():
#             for idx in np.where(mask)[0]:
#                 a_orig = A_np[idx, :]
#                 a_sign = np.sign(a_orig)
#                 a_abs = np.abs(a_orig)
#                 a = np.sort(a_abs)
#                 s = np.sum(a) - v
#                 l = float(len(a))
#                 for i in range(len(a)):
#                     if s / l > a[i]:
#                         s -= a[i]
#                         l -= 1
#                     else:
#                         break
#                 alpha = s / l
#                 a = a_sign * np.maximum(a_abs - alpha, 0)
#                 A_np[idx, :] = a
#             A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))

#         return super(ImplicitFunctionInf, cls).forward(ctx, A, B, X0, U, max_iter, tol)
    
#     @classmethod
#     def backward(cls, ctx, *grad_outputs):
#         grads = super(ImplicitFunctionInf, cls).backward(ctx, *grad_outputs)
#         # Add None for spectral_norm argument
#         return grads + (None,)


# class ImplicitModel(nn.Module):
#     def __init__(self, n: int, p: int, q: int,
#                  f: Optional[ImplicitFunction] = ImplicitFunctionInf,
#                  no_D: Optional[bool] = False,
#                  bias: Optional[bool] = False,
#                  max_iter: int = 1000,
#                  tol: float = 1e-5,
#                  spectral_norm: float = 0.9):
#         super(ImplicitModel, self).__init__()
#         if bias:
#             p += 1
#         self.n = n
#         self.p = p
#         self.q = q
#         self.max_iter = max_iter
#         self.tol = tol
#         self.spectral_norm = spectral_norm

#         std = 1.0 / np.sqrt(n)
#         self.A = nn.Parameter(torch.randn(n, n) * std)
#         self.B = nn.Parameter(torch.randn(n, p) * std)
#         self.C = nn.Parameter(torch.randn(q, n) * std)
#         self.D = nn.Parameter(torch.randn(q, p) * std) if not no_D else torch.zeros((q, p), requires_grad=False)
#         self.f = f
#         self.bias = bias

#     def forward(self, U: torch.Tensor, X0: Optional[torch.Tensor] = None):
#         U = transpose(U)
#         if self.bias:
#             U = F.pad(U, (0, 0, 0, 1), value=1)
        
#         m = U.shape[1]
#         X_shape = torch.Size([self.n, m])

#         if X0 is not None:
#             X0 = transpose(X0)
#         else:
#             X0 = torch.zeros(X_shape, dtype=U.dtype, device=U.device)

#         if self.f == ImplicitFunctionInf:
#             X = self.f.apply(self.A, self.B, X0, U, self.max_iter, self.tol, self.spectral_norm)
#         else:
#             X = self.f.apply(self.A, self.B, X0, U, self.max_iter, self.tol)
        
#         return transpose(self.C @ X + self.D @ U)


# class ImplicitRNNCell(nn.Module):
#     def __init__(self, input_size, n, hidden_size, max_iter=1000, tol=1e-5, spectral_norm=0.9, **kwargs):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.layer = ImplicitModel(
#             n, input_size + hidden_size, hidden_size, 
#             max_iter=max_iter, tol=tol, spectral_norm=spectral_norm, 
#             **kwargs
#         )

#     def forward(self, x, hx=None):
#         batch_size, seq_len, _ = x.shape
#         outputs = torch.empty(batch_size, seq_len, self.hidden_size, device=x.device)
        
#         if hx is None:
#             h = torch.zeros(batch_size, self.hidden_size, device=x.device)
#         else:
#             h = hx[0] if isinstance(hx, (tuple, list)) else hx

#         for t in range(seq_len):
#             combined_input = torch.cat((x[:, t, :], h), dim=-1)
#             h = self.layer(combined_input)
#             outputs[:, t, :] = h
            
#         return outputs, h

# # -------------------------------------------------------------------------
# # TempFlow Network
# # -------------------------------------------------------------------------

# class TempFlowTrainingNetwork(nn.Module):
#     @validated()
#     def __init__(
#         self,
#         input_size: int,
#         num_layers: int,
#         num_cells: int,
#         cell_type: str,
#         history_length: int,
#         context_length: int,
#         prediction_length: int,
#         dropout_rate: float,
#         lags_seq: List[int],
#         target_dim: int,
#         conditioning_length: int,
#         flow_type: str,
#         n_blocks: int,
#         hidden_size: int,
#         n_hidden: int,
#         dequantize: bool,
#         cardinality: List[int] = [1],
#         embedding_dimension: int = 1,
#         scaling: bool = True,
#         # IDL Parameters
#         idl_internal_dim: Optional[int] = None, 
#         idl_max_iter: int = 1000,
#         idl_tol: float = 1e-5,
#         idl_spectral_norm: float = 0.9,
#         **kwargs,
#     ) -> None:
#         super().__init__(**kwargs)
#         self.target_dim = target_dim
#         self.prediction_length = prediction_length
#         self.context_length = context_length
#         self.history_length = history_length
#         self.scaling = scaling
#         lags_seq.sort()
#         self.lags_seq = lags_seq
#         self.cell_type = cell_type
        
#         if cell_type == "Implicit":
#             # If internal dim 'n' is not specified, default to recurrent dim 'num_cells'
#             n_dim = idl_internal_dim if idl_internal_dim is not None else num_cells
            
#             self.rnn = ImplicitRNNCell(
#                 input_size=input_size,
#                 n=n_dim,               
#                 hidden_size=num_cells, 
#                 max_iter=idl_max_iter,
#                 tol=idl_tol,
#                 spectral_norm=idl_spectral_norm
#             )
#         else:
#             rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
#             self.rnn = rnn_cls(
#                 input_size=input_size,
#                 hidden_size=num_cells,
#                 num_layers=num_layers,
#                 dropout=dropout_rate,
#                 batch_first=True,
#             )

#         flow_cls = {"RealNVP": RealNVP, "MAF": MAF}[flow_type]
#         self.flow = flow_cls(
#             input_size=target_dim,
#             n_blocks=n_blocks,
#             n_hidden=n_hidden,
#             hidden_size=hidden_size,
#             cond_label_size=conditioning_length,
#         )
#         self.dequantize = dequantize
#         self.distr_output = FlowOutput(self.flow, input_size=target_dim, cond_size=conditioning_length)
#         self.proj_dist_args = self.distr_output.get_args_proj(num_cells)
#         self.embed_dim = 1
#         self.embed = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.embed_dim)
#         if self.scaling:
#             self.scaler = MeanScaler(keepdim=True)
#         else:
#             self.scaler = NOPScaler(keepdim=True)

#     @staticmethod
#     def get_lagged_subsequences(sequence, sequence_length, indices, subsequences_length=1):
#         assert max(indices) + subsequences_length <= sequence_length, "lags cannot go further than history length"
#         assert all(lag_index >= 0 for lag_index in indices)
#         lagged_values = []
#         for lag_index in indices:
#             begin_index = -lag_index - subsequences_length
#             end_index = -lag_index if lag_index > 0 else None
#             lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
#         return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

#     def unroll(self, lags, scale, time_feat, target_dimension_indicator, unroll_length, begin_state=None):
#         lags_scaled = lags / scale.unsqueeze(-1)
#         input_lags = lags_scaled.reshape((-1, unroll_length, len(self.lags_seq) * self.target_dim))
#         index_embeddings = self.embed(target_dimension_indicator)
#         repeated_index_embeddings = index_embeddings.unsqueeze(1).expand(-1, unroll_length, -1, -1).reshape((-1, unroll_length, self.target_dim * self.embed_dim))
#         inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1)
#         outputs, state = self.rnn(inputs, begin_state)
#         return outputs, state, lags_scaled, inputs

#     def unroll_encoder(self, past_time_feat, past_target_cdf, past_observed_values, past_is_pad, future_time_feat, future_target_cdf, target_dimension_indicator):
#         past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))
#         if future_time_feat is None or future_target_cdf is None:
#             time_feat = past_time_feat[:, -self.context_length :, ...]
#             sequence = past_target_cdf
#             sequence_length = self.history_length
#             subsequences_length = self.context_length
#         else:
#             time_feat = torch.cat((past_time_feat[:, -self.context_length :, ...], future_time_feat), dim=1)
#             sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
#             sequence_length = self.history_length + self.prediction_length
#             subsequences_length = self.context_length + self.prediction_length

#         lags = self.get_lagged_subsequences(sequence=sequence, sequence_length=sequence_length, indices=self.lags_seq, subsequences_length=subsequences_length)
#         _, scale = self.scaler(past_target_cdf[:, -self.context_length :, ...], past_observed_values[:, -self.context_length :, ...])
#         outputs, states, lags_scaled, inputs = self.unroll(lags=lags, scale=scale, time_feat=time_feat, target_dimension_indicator=target_dimension_indicator, unroll_length=subsequences_length, begin_state=None)
#         return outputs, states, scale, lags_scaled, inputs

#     def distr_args(self, rnn_outputs):
#         (distr_args,) = self.proj_dist_args(rnn_outputs)
#         return distr_args

#     def forward(self, target_dimension_indicator, past_time_feat, past_target_cdf, past_observed_values, past_is_pad, future_time_feat, future_target_cdf, future_observed_values):
#         rnn_outputs, _, scale, _, _ = self.unroll_encoder(past_time_feat, past_target_cdf, past_observed_values, past_is_pad, future_time_feat, future_target_cdf, target_dimension_indicator)
#         target = torch.cat((past_target_cdf[:, -self.context_length :, ...], future_target_cdf), dim=1)
#         distr_args = self.distr_args(rnn_outputs)
#         if self.scaling: self.flow.scale = scale
#         if self.dequantize: target += torch.rand_like(target)
#         likelihoods = -self.flow.log_prob(target, distr_args).unsqueeze(-1)
#         past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))
#         observed_values = torch.cat((past_observed_values[:, -self.context_length :, ...], future_observed_values), dim=1)
#         loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
#         loss = weighted_average(likelihoods, weights=loss_weights, dim=1)
#         return (loss.mean(), likelihoods, distr_args)


# class TempFlowPredictionNetwork(TempFlowTrainingNetwork):
#     def __init__(self, num_parallel_samples: int, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.num_parallel_samples = num_parallel_samples
#         self.shifted_lags = [l - 1 for l in self.lags_seq]

#     def sampling_decoder(self, past_target_cdf, target_dimension_indicator, time_feat, scale, begin_states):
#         def repeat(tensor, dim=0):
#             return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

#         repeated_past_target_cdf = repeat(past_target_cdf)
#         repeated_time_feat = repeat(time_feat)
#         repeated_scale = repeat(scale)
#         if self.scaling: self.flow.scale = repeated_scale
#         repeated_target_dimension_indicator = repeat(target_dimension_indicator)

#         # Handle different state shapes for Implicit (batch, hidden) vs LSTM/GRU (layers, batch, hidden)
#         if self.cell_type == "Implicit":
#             repeated_states = repeat(begin_states, dim=0)
#         elif self.cell_type == "LSTM":
#             repeated_states = [repeat(s, dim=1) for s in begin_states]
#         else:
#             # GRU
#             repeated_states = repeat(begin_states, dim=1)

#         future_samples = []
#         for k in range(self.prediction_length):
#             lags = self.get_lagged_subsequences(sequence=repeated_past_target_cdf, sequence_length=self.history_length + k, indices=self.shifted_lags, subsequences_length=1)
#             rnn_outputs, repeated_states, _, _ = self.unroll(begin_state=repeated_states, lags=lags, scale=repeated_scale, time_feat=repeated_time_feat[:, k : k + 1, ...], target_dimension_indicator=repeated_target_dimension_indicator, unroll_length=1)
#             distr_args = self.distr_args(rnn_outputs)
#             new_samples = self.flow.sample(cond=distr_args)
#             future_samples.append(new_samples)
#             repeated_past_target_cdf = torch.cat((repeated_past_target_cdf, new_samples), dim=1)

#         samples = torch.cat(future_samples, dim=1)
#         return samples.reshape((-1, self.num_parallel_samples, self.prediction_length, self.target_dim))

#     def forward(self, target_dimension_indicator, past_time_feat, past_target_cdf, past_observed_values, past_is_pad, future_time_feat):
#         past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))
#         _, begin_states, scale, _, _ = self.unroll_encoder(past_time_feat, past_target_cdf, past_observed_values, past_is_pad, None, None, target_dimension_indicator)
#         return self.sampling_decoder(past_target_cdf, target_dimension_indicator, future_time_feat, scale, begin_states)

from typing import List, Optional, Tuple, Union
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from gluonts.core.component import validated

from pts.model import weighted_average
from pts.modules import RealNVP, MAF, FlowOutput, MeanScaler, NOPScaler, EquilibriumFlow


# -------------------------------------------------------------------------
# Implicit Deep Learning Components (Sourced from alicia-tsai repo)
# -------------------------------------------------------------------------

def transpose(x):
    return x.transpose(-2, -1)

class ImplicitFunctionWarning(RuntimeWarning):
    pass

class ImplicitFunction(Function):
    @classmethod
    def forward(cls, ctx, A, B, X0, U, mitr, tol):
        ctx.mitr = mitr
        ctx.tol = tol
        with torch.no_grad():
            X, err, status = cls.inn_pred(A, B @ U, X0, mitr, tol)
        ctx.save_for_backward(A, B, X, U)
        return X

    @classmethod
    def backward(cls, ctx, *grad_outputs):
        A, B, X, U = ctx.saved_tensors
        grad_output = grad_outputs[0]
        
        # Ensure gradient size matches
        assert grad_output.size() == X.size()

        DPhi = cls.dphi(A @ X + B @ U)
        V, err, status = cls.inn_pred_grad(A.T, DPhi * grad_output, DPhi, ctx.mitr, ctx.tol)
        
        grad_A = V @ X.T
        grad_B = V @ U.T
        grad_U = B.T @ V

        # Return gradients for A, B, X0, U, mitr, tol
        return grad_A, grad_B, torch.zeros_like(X), grad_U, None, None

    @staticmethod
    def phi(X):
        return torch.clamp(X, min=0)

    @staticmethod
    def dphi(X):
        grad = X.clone().detach()
        grad[X <= 0] = 0
        grad[X > 0] = 1
        return grad

    @classmethod
    def inn_pred(cls, A, Z, X, mitr, tol):
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            X_new = cls.phi(A @ X + Z)
            err = torch.norm(X_new - X, float('inf'))
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status

    @staticmethod
    def inn_pred_grad(AT, Z, DPhi, mitr, tol):
        X = torch.zeros_like(Z)
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            X_new = DPhi * (AT @ X) + Z
            err = torch.norm(X_new - X, float('inf'))
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status


class ImplicitFunctionInf(ImplicitFunction):
    @classmethod
    def forward(cls, ctx, A, B, X0, U, mitr, tol, spectral_norm):
        v = spectral_norm
        A_np = A.clone().detach().cpu().numpy()
        x = np.abs(A_np).sum(axis=-1)
        mask = x > v
        
        if mask.any():
            for idx in np.where(mask)[0]:
                a_orig = A_np[idx, :]
                a_sign = np.sign(a_orig)
                a_abs = np.abs(a_orig)
                a = np.sort(a_abs)
                s = np.sum(a) - v
                l = float(len(a))
                for i in range(len(a)):
                    if s / l > a[i]:
                        s -= a[i]
                        l -= 1
                    else:
                        break
                alpha = s / l
                a = a_sign * np.maximum(a_abs - alpha, 0)
                A_np[idx, :] = a
            A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))

        return super(ImplicitFunctionInf, cls).forward(ctx, A, B, X0, U, mitr, tol)
    
    @classmethod
    def backward(cls, ctx, *grad_outputs):
        grads = super(ImplicitFunctionInf, cls).backward(ctx, *grad_outputs)
        # Pad with None for the spectral_norm argument
        return grads + (None,)


class ImplicitModel(nn.Module):
    def __init__(self, n: int, p: int, q: int,
                 f: Optional[ImplicitFunction] = ImplicitFunctionInf,
                 no_D: Optional[bool] = False,
                 bias: Optional[bool] = False,
                 max_iter: int = 500,
                 tol: float = 1e-4,
                 spectral_norm: float = 0.9):
        super(ImplicitModel, self).__init__()
        if bias:
            p += 1
        self.n = n
        self.p = p
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.spectral_norm = spectral_norm

        self.A = nn.Parameter(torch.randn(n, n)/n)
        self.B = nn.Parameter(torch.randn(n, p)/n)
        self.C = nn.Parameter(torch.randn(q, n)/n)
        self.D = nn.Parameter(torch.randn(q, p)/n) if not no_D else torch.zeros((q, p), requires_grad=False)
        self.f = f
        self.bias = bias

    def forward(self, U: torch.Tensor, X0: Optional[torch.Tensor] = None):
        U = transpose(U)
        if self.bias:
            U = F.pad(U, (0, 0, 0, 1), value=1)
        assert U.shape[0] == self.p, f'Given input size {U.shape[0]} does not match expected size {self.p}.'
        
        m = U.shape[1]
        X_shape = torch.Size([self.n, m])
        if X0 is not None:
            X0 = transpose(X0)
        else:
            X0 = torch.zeros(X_shape, dtype=U.dtype, device=U.device)

        if self.f == ImplicitFunctionInf:
            X = self.f.apply(self.A, self.B, X0, U, self.max_iter, self.tol, self.spectral_norm)
        else:
            X = self.f.apply(self.A, self.B, X0, U, self.max_iter, self.tol)
        
        return transpose(self.C @ X + self.D @ U)


class ImplicitRNNCell(nn.Module):
    def __init__(self, input_size, n, hidden_size, max_iter=500, tol=1e-4, spectral_norm=0.9, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer = ImplicitModel(
            n, input_size + hidden_size, hidden_size, 
            max_iter=max_iter, tol=tol, spectral_norm=spectral_norm, 
            **kwargs
        )
        # CRITICAL FIX 1: LayerNorm prevents the state from exploding/vanishing over 500+ unroll steps
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hx=None):
        batch_size, seq_len, _ = x.shape
        
        if hx is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = hx[0] if hx.dim() == 3 else hx

        outputs = []
        for t in range(seq_len):
            combined_input = torch.cat((x[:, t, :], h), dim=-1)
            h = self.layer(combined_input)
            
            # Apply Normalization to maintain stable distribution
            h = self.norm(h)
            outputs.append(h)
            
        # CRITICAL FIX 2: Use torch.stack to prevent in-place autograd errors
        outputs = torch.stack(outputs, dim=1) 
        
        # Mimics GRU shape: (num_layers=1, batch_size, hidden_size)
        return outputs, h.unsqueeze(0)


# -------------------------------------------------------------------------
# TempFlow Network Implementation
# -------------------------------------------------------------------------

class TempFlowTrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        dropout_rate: float,
        lags_seq: List[int],
        target_dim: int,
        conditioning_length: int,
        flow_type: str,
        n_blocks: int,
        hidden_size: int,
        n_hidden: int,
        dequantize: bool,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        scaling: bool = True,
        # Configurable IDL Parameters
        idl_internal_dim: Optional[int] = None, 
        idl_max_iter: int = 500,
        idl_tol: float = 1e-4,
        idl_spectral_norm: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling

        assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq

        self.cell_type = cell_type
        
        if cell_type == "Implicit":
            n_dim = idl_internal_dim if idl_internal_dim is not None else num_cells
            self.rnn = ImplicitRNNCell(
                input_size=input_size,
                n=n_dim,               
                hidden_size=num_cells, 
                max_iter=idl_max_iter,
                tol=idl_tol,
                spectral_norm=idl_spectral_norm
            )
        else:
            rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
            self.rnn = rnn_cls(
                input_size=input_size,
                hidden_size=num_cells,
                num_layers=num_layers,
                dropout=dropout_rate,
                batch_first=True,
            )

        flow_cls = {"RealNVP": RealNVP, "MAF": MAF, "Equilibrium": EquilibriumFlow}[flow_type]
        if flow_type == "Equilibrium":
            self.flow = flow_cls(
                input_size=target_dim,
                hidden_size=hidden_size, # Maps to your idl_internal_dim
                n_blocks=n_blocks,
                cond_label_size=conditioning_length,
                idl_max_iter=kwargs.get("idl_max_iter", 1000),
                idl_tol=kwargs.get("idl_tol", 3e-6),
                idl_spectral_norm=kwargs.get("idl_spectral_norm", 0.95)
            )
        else:
            self.flow = flow_cls(
                input_size=target_dim,
                n_blocks=n_blocks,
                n_hidden=n_hidden,
                hidden_size=hidden_size,
                cond_label_size=conditioning_length,
            )

        self.dequantize = dequantize
        self.distr_output = FlowOutput(self.flow, input_size=target_dim, cond_size=conditioning_length)
        self.proj_dist_args = self.distr_output.get_args_proj(num_cells)
        self.embed_dim = 1
        self.embed = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.embed_dim)
        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(sequence, sequence_length, indices, subsequences_length=1):
        assert max(indices) + subsequences_length <= sequence_length, "lags cannot go further than history length"
        assert all(lag_index >= 0 for lag_index in indices)
        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def unroll(self, lags, scale, time_feat, target_dimension_indicator, unroll_length, begin_state=None):
        lags_scaled = lags / scale.unsqueeze(-1)
        input_lags = lags_scaled.reshape((-1, unroll_length, len(self.lags_seq) * self.target_dim))
        index_embeddings = self.embed(target_dimension_indicator)
        repeated_index_embeddings = index_embeddings.unsqueeze(1).expand(-1, unroll_length, -1, -1).reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1)
        outputs, state = self.rnn(inputs, begin_state)
        return outputs, state, lags_scaled, inputs

    def unroll_encoder(self, past_time_feat, past_target_cdf, past_observed_values, past_is_pad, future_time_feat, future_target_cdf, target_dimension_indicator):
        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))
        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat((past_time_feat[:, -self.context_length :, ...], future_time_feat), dim=1)
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        lags = self.get_lagged_subsequences(sequence=sequence, sequence_length=sequence_length, indices=self.lags_seq, subsequences_length=subsequences_length)
        _, scale = self.scaler(past_target_cdf[:, -self.context_length :, ...], past_observed_values[:, -self.context_length :, ...])
        outputs, states, lags_scaled, inputs = self.unroll(lags=lags, scale=scale, time_feat=time_feat, target_dimension_indicator=target_dimension_indicator, unroll_length=subsequences_length, begin_state=None)
        return outputs, states, scale, lags_scaled, inputs

    def distr_args(self, rnn_outputs):
        (distr_args,) = self.proj_dist_args(rnn_outputs)
        return distr_args

    def forward(self, target_dimension_indicator, past_time_feat, past_target_cdf, past_observed_values, past_is_pad, future_time_feat, future_target_cdf, future_observed_values):
        rnn_outputs, _, scale, _, _ = self.unroll_encoder(past_time_feat, past_target_cdf, past_observed_values, past_is_pad, future_time_feat, future_target_cdf, target_dimension_indicator)
        target = torch.cat((past_target_cdf[:, -self.context_length :, ...], future_target_cdf), dim=1)
        distr_args = self.distr_args(rnn_outputs)
        if self.scaling: self.flow.scale = scale
        if self.dequantize: target += torch.rand_like(target)
        likelihoods = -self.flow.log_prob(target, distr_args).unsqueeze(-1)
        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))
        observed_values = torch.cat((past_observed_values[:, -self.context_length :, ...], future_observed_values), dim=1)
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(likelihoods, weights=loss_weights, dim=1)
        return (loss.mean(), likelihoods, distr_args)


class TempFlowPredictionNetwork(TempFlowTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(self, past_target_cdf, target_dimension_indicator, time_feat, scale, begin_states):
        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        if self.scaling: self.flow.scale = repeated_scale
        repeated_target_dimension_indicator = repeat(target_dimension_indicator)

        if self.cell_type == "LSTM":
            repeated_states = [repeat(s, dim=1) for s in begin_states]
        else:
            # Both GRU and Implicit now use shape (1, batch, hidden)
            repeated_states = repeat(begin_states, dim=1)

        future_samples = []
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(sequence=repeated_past_target_cdf, sequence_length=self.history_length + k, indices=self.shifted_lags, subsequences_length=1)
            rnn_outputs, repeated_states, _, _ = self.unroll(begin_state=repeated_states, lags=lags, scale=repeated_scale, time_feat=repeated_time_feat[:, k : k + 1, ...], target_dimension_indicator=repeated_target_dimension_indicator, unroll_length=1)
            distr_args = self.distr_args(rnn_outputs)
            new_samples = self.flow.sample(cond=distr_args)
            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat((repeated_past_target_cdf, new_samples), dim=1)

        samples = torch.cat(future_samples, dim=1)
        return samples.reshape((-1, self.num_parallel_samples, self.prediction_length, self.target_dim))

    def forward(self, target_dimension_indicator, past_time_feat, past_target_cdf, past_observed_values, past_is_pad, future_time_feat):
        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))
        _, begin_states, scale, _, _ = self.unroll_encoder(past_time_feat, past_target_cdf, past_observed_values, past_is_pad, None, None, target_dimension_indicator)
        return self.sampling_decoder(past_target_cdf, target_dimension_indicator, future_time_feat, scale, begin_states)
        