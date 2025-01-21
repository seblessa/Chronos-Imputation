from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import numpy as np
import logging
import imputers
import warnings
import utils
import yaml
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    
    # Suppress Autogluon logs
    logging.getLogger('autogluon').setLevel(logging.WARNING)

    # Set the root logger level
    logging.basicConfig(level=logging.WARNING)
        
    MODELS = [
        GradientBoostingRegressor(random_state=42),
        RandomForestRegressor(random_state=42)
    ]
    
    IMPUTATION_METHODS = ['interpolation', 'knn', 'gmm', 'chronos']
    
    MODES = {
        "20% of data removed in random rows": None,
        "20% of data removed in 5 sequences": (0.2, 5),
        "20% of data removed in 1 sequence": (0.2, 1),
    }
        
    dataset_getters = [
        (utils.load_dataset_etth,"etth"),
        # add more datasets here
    ]
    
    for get_dataset, name in dataset_getters:
        df, DATETIME, DATETIME_FORMAT, TARGET = get_dataset()
    
        reg_results = utils.evaluate_time_series_regression(df, DATETIME, DATETIME_FORMAT, TARGET, MODELS, verbose=False)
        
        # Initialize results dictionary
        results = {
            "datasets": {
                "original": (df, DATETIME, DATETIME_FORMAT, name),
                "imputed": {mode: {method: None for method in IMPUTATION_METHODS} for mode in MODES},
            },
            "values": {
                "regression_errors": {"original": {"mae": reg_results['mae'], "rmse": reg_results['rmse']},
                                    **{mode: {method: {"mae": None, "rmse": None} for method in IMPUTATION_METHODS} for mode in MODES}},
                "regression_r2_score": {"original": reg_results['r2'], **{mode: {method: None for method in IMPUTATION_METHODS} for mode in MODES}},
                "imputation_errors": {"ac_score_original": {mode: None for mode in MODES},
                                    **{mode: {method: {"nmae": None, "nrmse": None, "ac_score_imputed": None} for method in IMPUTATION_METHODS} for mode in MODES}},
            },
        }
        
        # Mean runtime of each imputation method
        runtimes = {method: [] for method in IMPUTATION_METHODS}

        # Introduce and evaluate missing values
        for mode, params in MODES.items():
            logger.info(f"Processing mode: {mode}")

            if mode == "20% of data removed in random rows":
                df_nan, nan_positions = utils.introduce_nans_randomly(df, 0.2, DATETIME, TARGET)
            else:
                df_nan, nan_positions = utils.introduce_nans_sequentially(df, *params, DATETIME, TARGET)

            for method, imputer_func in {
                'interpolation': imputers.impute_with_interpolation,
                'knn': imputers.impute_with_knn,
                'gmm': imputers.impute_with_gmm,
                'chronos': imputers.impute_with_chronos,
            }.items():
                logger.info(f"Using imputer: {method}")
                
                start_time = pd.Timestamp.now()
                temp_df = imputer_func(df_nan, DATETIME, DATETIME_FORMAT)
                end_time = pd.Timestamp.now()
                
                runtime = (end_time - start_time).total_seconds()
                runtimes[method].append(runtime)
                
                results["datasets"]["imputed"][mode][method] = (temp_df, DATETIME, DATETIME_FORMAT, name)

                # Regression evaluation
                reg_results = utils.evaluate_time_series_regression(temp_df, DATETIME, DATETIME_FORMAT, TARGET, MODELS, verbose=False)
                results["values"]["regression_errors"][mode][method].update({"mae": reg_results['mae'], "rmse": reg_results['rmse']})
                results["values"]["regression_r2_score"][mode][method] = reg_results['r2']

                # Imputation evaluation
                imp_results = utils.evaluate_imputation(df, temp_df, nan_positions, DATETIME, DATETIME_FORMAT, verbose=False)
                results["values"]["imputation_errors"]["ac_score_original"][mode]= imp_results['ac_score_original']
                results["values"]["imputation_errors"][mode][method].update({"nmae": imp_results['nmae'], "nrmse": imp_results['nrmse'], "ac_score_imputed": imp_results['ac_score_imputed']})

            logger.info("Mode processing completed\n")

        # Calculate mean runtimes
        runtimes = {method: round(float(np.mean(runtimes[method])), 3) for method in IMPUTATION_METHODS}

        utils.save_results(results, runtimes, name)
        
        print(f"Results for {name} dataset saved\n")
