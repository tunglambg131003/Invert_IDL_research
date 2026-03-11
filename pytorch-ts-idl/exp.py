import numpy as np
import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

from pts.model.tempflow import TempFlowEstimator
from pts import Trainer

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data Set
    print("Loading solar_nips dataset...")
    try:
        dataset = get_dataset("solar_nips", regenerate=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Group the time series for multivariate training
    target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
    train_grouper = MultivariateGrouper(max_target_dim=target_dim)
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)), 
        max_target_dim=target_dim
    )

    dataset_train = train_grouper(dataset.train) * 100
    dataset_test = test_grouper(dataset.test)

    # 3. Setup Evaluator
    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(20)/20.0)[1:],
        target_agg_funcs={'sum': np.sum}
    )

    # 4. Instantiate TempFlow Estimator with Implicit Deep Learning
    print("Initializing TempFlowEstimator with cell_type='Implicit'...")
    estimator = TempFlowEstimator(
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        prediction_length=dataset.metadata.prediction_length,
        cell_type='Implicit', 
        
        # --- Configurable IDL Parameters ---
        num_cells=64,            
        hidden_size=300,         
        idl_max_iter=500,       
        idl_tol=1e-6,           
        idl_spectral_norm=0.95,  

        # -----------------------------------

        input_size=552,
        freq=dataset.metadata.freq,
        scaling=True,
        dequantize=True,
        n_blocks=4,
        trainer=Trainer(
            device=device,
            epochs=40,
            learning_rate=1e-3,
            num_batches_per_epoch=100,
            batch_size=64,
        )
    )

    # 5. Train the Model
    print("Starting training...")
    predictor = estimator.train(
        dataset_train, 
        num_workers=0,          
        prefetch_factor=None    
    )

    # 6. Make Predictions
    print("Generating predictions...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test,
        predictor=predictor,
        num_samples=100
    )
    forecasts = list(forecast_it)
    targets = list(ts_it)

    # 7. Evaluate
    print("Evaluating results...")
    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

    # 8. Print Metrics
    print("\n--- Model Metrics ---")
    print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
    print("ND: {}".format(agg_metric['ND']))
    print("NRMSE: {}".format(agg_metric['NRMSE']))
    print("MSE: {}".format(agg_metric['MSE']))

    print("\n--- Summation Metrics ---")
    print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
    print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
    print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
    print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))

if __name__ == "__main__":
    main()
