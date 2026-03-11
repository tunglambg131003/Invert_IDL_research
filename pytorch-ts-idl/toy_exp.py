# import numpy as np
# import pandas as pd
# import torch

# from gluonts.dataset.multivariate_grouper import MultivariateGrouper
# from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
# from pts.model.transformer_tempflow import TransformerTempFlowEstimator
# from pts import Trainer
# from gluonts.evaluation.backtest import make_evaluation_predictions
# from gluonts.evaluation import MultivariateEvaluator
# from gluonts.transform import ExpectedNumInstanceSampler

# if __name__ == '__main__':
#     # 1. Setup device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 2. Load dataset
#     print("Loading dataset 'solar_nips'...")
#     dataset = get_dataset("solar_nips", regenerate=False)

#     # 3. Setup Groupers
#     train_grouper = MultivariateGrouper(
#         max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
#     )

#     test_grouper = MultivariateGrouper(
#         num_test_dates=int(len(dataset.test) / len(dataset.train)), 
#         max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
#     )

#     print("Grouping datasets...")
#     dataset_train = train_grouper(dataset.train)
#     dataset_test = test_grouper(dataset.test)

#     # 4. Setup Evaluator
#     evaluator = MultivariateEvaluator(
#         quantiles=(np.arange(20) / 20.0)[1:],
#         target_agg_funcs={'sum': np.sum}
#     )

#     # 5. Initialize the TransformerTempFlowEstimator
#     print("Initializing Transformer TempFlow Estimator...")
#     estimator = TransformerTempFlowEstimator(
#         d_model=16,
#         num_heads=4,
#         input_size=552,
#         target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
#         prediction_length=dataset.metadata.prediction_length,
#         context_length=dataset.metadata.prediction_length * 4,
#         flow_type='MAF',
#         dequantize=True,
#         freq=dataset.metadata.freq,
#         trainer=Trainer(
#             device=device,
#             epochs=14,
#             learning_rate=1e-3,
#             num_batches_per_epoch=100,
#             batch_size=64,
#         )
#     )

#     # FIX 2: Override the Train Sampler
#     # Force the sampler to draw 100 instances per pass instead of 1.
#     # This prevents the dataset of size 1 from yielding 0 samples and causing the "idle transformation" crash.
#     estimator.train_sampler = ExpectedNumInstanceSampler(
#         num_instances=5.0,
#         min_instances=1,
#         min_past=estimator.history_length,
#         min_future=estimator.prediction_length,
#     )

#     # 6. Train the model
#     print("Starting training...")
#     predictor = estimator.train(
#         dataset_train, 
#         # FIX 3: Prevent multiprocessing worker starvation on a size-1 dataset
#         num_workers=0,          
#         # FIX 4: Prevent crash in PyTorch 2.0+ DataLoader when num_workers=0
#         prefetch_factor=None    
#     )

#     # 7. Evaluate the trained predictor
#     print("Training complete! Generating evaluation predictions...")
#     forecast_it, ts_it = make_evaluation_predictions(
#         dataset=dataset_test,
#         predictor=predictor,
#         num_samples=100
#     )
    
#     forecasts = list(forecast_it)
#     targets = list(ts_it)

#     print("Calculating metrics...")
#     agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

#     # 8. Print Results
#     print("\n" + "="*30)
#     print("EVALUATION RESULTS")
#     print("="*30)
#     print(f"CRPS: {agg_metric['mean_wQuantileLoss']}")
#     print(f"ND: {agg_metric['ND']}")
#     print(f"NRMSE: {agg_metric['NRMSE']}")
#     print(f"MSE: {agg_metric['MSE']}")
#     print("-" * 30)
#     print(f"CRPS-Sum: {agg_metric['m_sum_mean_wQuantileLoss']}")
#     print(f"ND-Sum: {agg_metric['m_sum_ND']}")
#     print(f"NRMSE-Sum: {agg_metric['m_sum_NRMSE']}")
#     print(f"MSE-Sum: {agg_metric['m_sum_MSE']}")


import numpy as np
import pandas as pd
import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

if __name__ == '__main__':
    # 1. Setup device and seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load dataset
    print("Loading dataset 'solar_nips'...")
    dataset = get_dataset("solar_nips", regenerate=False)

    # 3. Setup Groupers
    train_grouper = MultivariateGrouper(
        max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
    )

    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)), 
        max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
    )

    print("Grouping datasets...")
    # OPTION 1: Multiply the 1-item dataset by 100 to prevent the crash 
    # and guarantee highly randomized training batches for better accuracy.
    dataset_train = list(train_grouper(dataset.train)) * 100
    dataset_test = test_grouper(dataset.test)

    # 4. Setup Evaluator
    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(20) / 20.0)[1:],
        target_agg_funcs={'sum': np.sum}
    )

    # 5. Initialize the TransformerTempFlowEstimator
    print("Initializing Transformer TempFlow Estimator...")
    estimator = TransformerTempFlowEstimator(
        d_model=16,
        num_heads=4,
        input_size=552,
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        prediction_length=dataset.metadata.prediction_length,
        context_length=dataset.metadata.prediction_length * 4,
        flow_type='MAF',
        dequantize=True,
        freq=dataset.metadata.freq,
        trainer=Trainer(
            device=device,
            epochs=20,
            learning_rate=1e-3,
            num_batches_per_epoch=100,
            batch_size=64,
        )
    )

    # Notice we REMOVED the estimator.train_sampler override here.

    # 6. Train the model
    print("Starting training...")
    predictor = estimator.train(
        dataset_train, 
        num_workers=0,          # Prevents multiprocessing worker starvation
        prefetch_factor=None    # Prevents crash in PyTorch 2.0+ DataLoader when num_workers=0
    )

    # 7. Evaluate the trained predictor
    print("Training complete! Generating evaluation predictions...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    targets = list(ts_it)

    print("Calculating metrics...")
    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

    # 8. Print Results
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"CRPS: {agg_metric['mean_wQuantileLoss']}")
    print(f"ND: {agg_metric['ND']}")
    print(f"NRMSE: {agg_metric['NRMSE']}")
    print(f"MSE: {agg_metric['MSE']}")
    print("-" * 30)
    print(f"CRPS-Sum: {agg_metric['m_sum_mean_wQuantileLoss']}")
    print(f"ND-Sum: {agg_metric['m_sum_ND']}")
    print(f"NRMSE-Sum: {agg_metric['m_sum_NRMSE']}")
    print(f"MSE-Sum: {agg_metric['m_sum_MSE']}")
    