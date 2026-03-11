# PyTorchTS

PyTorchTS is a [PyTorch](https://github.com/pytorch/pytorch) Probabilistic Time Series forecasting framework which provides state of the art PyTorch time series models by utilizing [GluonTS](https://github.com/awslabs/gluon-ts) as its back-end API and for loading, transforming and back-testing time series data sets.

## Installation

```
$ pip3 install pytorchts
```

## Example

You can see a sample run in `exp.py`.

This repository extends the original framework by introducing **Implicit Deep Learning components** into the **TempFlow model**. My modification is in the directory `pts/model/tempflow`, and I'm currently working to create IDL for flow_type model (`pts/modules/flows.py`)

# ImplicitRNN Cell

A new recurrent cell type has been introduced:

```
cell_type = "Implicit"
```

This adds **ImplicitRNN** as an alternative to standard recurrent cells.

Available options now include:

- `LSTM`
- `GRU`
- `Implicit` (ImplicitRNN)
