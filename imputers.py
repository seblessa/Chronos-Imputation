from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from multiprocessing import Manager, Pool
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from rich.live import Live
from queue import Empty
import pandas as pd
import numpy as np
import shutil
import time
import os


def _process_feature(args):
    feature, freq, max_prediction_length, random_state, finetune, datetime_col, verbose, update_queue, item_id = args
    feature_name = feature['name']
    path = f"/tmp/chronos_forecaster_{feature_name}_{time.time()}"

    try:
        predictor = TimeSeriesPredictor(
            target='target',
            prediction_length=max_prediction_length,
            known_covariates_names=list(feature['covariates'].keys()),
            freq=freq,
            path=path,
            cache_predictions=False,
            verbosity=0
        )

        ts_data = TimeSeriesDataFrame.from_data_frame(
            feature['data'], id_column='item_id', timestamp_column='timestamp'
        ).fill_missing_values(method="auto").ffill().bfill()

        predictor.fit(
            train_data=ts_data,
            presets="best_quality",
            hyperparameters={
                'Chronos': {
                    'model_path': 'autogluon/chronos-bolt-base',
                    'fine_tune': finetune,
                    'batch_size': 16
                }
            },
            random_seed=random_state,
            verbosity=0
        )

        data = feature['data']
        covariates = feature['covariates']
        
        # Fill the first 10 NaNs with the mean of the first 100 non-NaN values to avoid empty context data
        data.iloc[:10, data.columns.get_loc('target')] = data['target'].iloc[:10].fillna(data['target'].dropna().iloc[:min(100, len(data))].mean())

        # Iteratively fill missing values in blocks
        nan_indices = data.index[data['target'].isna()]
        while len(nan_indices) > 0:
            first_nan_index = nan_indices[0]
            context_start = max(0, first_nan_index - max_prediction_length * 5)
            context_data = data.iloc[context_start:first_nan_index]

            if context_data.empty:
                raise ValueError("Context data is unexpectedly empty. Check input data or previous predictions.")

            # Calculate the size of the NaN block to fill
            nan_block_size = min(len(nan_indices), max_prediction_length)

            # Predict for the NaN block
            prediction = predictor.predict(context_data, known_covariates=covariates, random_seed=random_state)

            # Fill only the required number of NaNs
            for i, idx in enumerate(nan_indices[:nan_block_size]):
                data.at[idx, 'target'] = prediction['mean'].values[i]

            # Update remaining NaN indices
            nan_indices = data.index[data['target'].isna()]

            if verbose:
                update_queue.put((item_id, nan_block_size))

        if verbose:
            update_queue.put((item_id, "done"))

        return data.rename(columns={'target': feature['name'], 'timestamp': datetime_col})[[datetime_col, feature['name']]]

    finally:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            
def impute_with_chronos(data: pd.DataFrame, datetime_col: str, datetime_format: str = None, max_prediction_length: int = 60,
                        verbose: bool = False, random_state: int = None, finetune: bool = False) -> pd.DataFrame:
    def prepare_data(data, datetime_col, feature):
        data_copy = data[[datetime_col, feature]].copy()
        data_copy = data_copy.rename(columns={feature: 'target', datetime_col: 'timestamp'})
        data_copy['item_id'] = 0
        covariates = TimeSeriesDataFrame.from_data_frame(
            data_copy.drop(columns=['target']), id_column='item_id', timestamp_column='timestamp'
        ).fill_missing_values(method="auto")
        return {'name': feature, 'data': data_copy, 'covariates': covariates}

    if datetime_format:
        data[datetime_col] = pd.to_datetime(data[datetime_col], format=datetime_format)
    else:
        data[datetime_col] = pd.to_datetime(data[datetime_col])

    freq = data[datetime_col].diff().mode()[0]
    features = [prepare_data(data, datetime_col, col) for col in data.columns if col != datetime_col]

    with Manager() as manager:
        update_queue = manager.Queue()
        try:
            if verbose:
                progress = Progress(
                    "[progress.description]{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeRemainingColumn(),
                )

                outer_task_id = progress.add_task("[blue]Overall Progress", total=len(features))
                task_ids = [progress.add_task(f"[cyan]Column {feature['name']}:", total=feature['data'].isna().sum().sum()) for feature in features]

                with Live(progress, refresh_per_second=5):
                    with Pool() as pool:
                        results = pool.map_async(
                            _process_feature,
                            [(feature, freq, max_prediction_length, random_state, finetune, datetime_col, verbose, update_queue, i) for i, feature in enumerate(features)]
                        )

                        while not progress.tasks[outer_task_id].finished:
                            try:
                                item_id, update = update_queue.get(timeout=0.1)
                                if update == "done":
                                    progress.remove_task(task_ids[item_id])
                                    progress.advance(outer_task_id, 1)
                                else:
                                    progress.advance(task_ids[item_id], update)
                            except Empty:
                                continue

                        # Properly close the pool and wait for tasks to finish
                        pool.close()
                        pool.join()

            else:
                with Pool() as pool:
                    results = pool.map_async(
                        _process_feature,
                        [(feature, freq, max_prediction_length, random_state, finetune, datetime_col, verbose, update_queue, i) for i, feature in enumerate(features)]
                    )
                    results.wait()

                    # Properly close the pool
                    pool.close()
                    pool.join()

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected! Terminating pool...")
            pool.terminate()
            pool.join()

    combined_context = pd.DataFrame()

    for result in results.get():
        combined_context = combined_context.merge(result, on=datetime_col, how='left') if not combined_context.empty else result

    return combined_context

def prepare_data(data: pd.DataFrame, datetime_col: str, datetime_format: str) -> pd.DataFrame:
    data = data.copy()
    
    if data[datetime_col].isnull().any():
        raise ValueError("Missing values found in the datetime column. Please handle missing datetimes before proceeding.")
    
    if datetime_col in data.columns:
        if datetime_format is not None:
            data[datetime_col] = pd.to_datetime(data[datetime_col], format=datetime_format)
        else:
            data[datetime_col] = pd.to_datetime(data[datetime_col])
        
        return data.set_index(datetime_col)
    else:
        raise ValueError("Datetime column not found in the DataFrame.")

def impute_with_gmm(df_nan: pd.DataFrame, datetime_col: str, datetime_format: str) -> pd.DataFrame:
    df_imputed = prepare_data(df_nan, datetime_col, datetime_format)

    for column in df_imputed.columns:
        if df_imputed[column].isnull().any():
            observed_data = df_imputed[column].dropna().values.reshape(-1, 1)
            missing_indices = df_imputed[column].isnull()
            n_missing = missing_indices.sum()
            best_gmm = None
            best_bic = np.inf
            for n_components in range(1, 10):
                gmm_candidate = GaussianMixture(n_components=n_components, random_state=42)
                gmm_candidate.fit(observed_data)
                bic = gmm_candidate.bic(observed_data)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm_candidate
            
            imputed_values_gmm = best_gmm.sample(n_missing)[0].flatten()
            df_imputed.loc[missing_indices, column] = imputed_values_gmm
            
    return df_imputed.reset_index()

def impute_with_knn(df_nan: pd.DataFrame, datetime_col: str, datetime_format: str) -> pd.DataFrame:
    """
    Impute missing values in a DataFrame using KNN imputation.
    
    Parameters:
        df_nan (pd.DataFrame): DataFrame with missing values.
        datetime_col (str): The name of the datetime column.
        datetime_format (str): The format of the datetime column.
        n_neighbors (int): Number of neighbors.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    
    df_imputed = prepare_data(df_nan, datetime_col, datetime_format)
    
    imputer = KNNImputer()
    df_imputed = pd.DataFrame(imputer.fit_transform(df_imputed), columns=df_imputed.columns, index=df_imputed.index)
    
    return df_imputed.reset_index()

def impute_with_interpolation(df_nan: pd.DataFrame, datetime_col: str, datetime_format: str) -> pd.DataFrame:
    """
    Impute missing values in a DataFrame using linear interpolation.
    
    Parameters:
        df_nan (pd.DataFrame): DataFrame with missing values.
        datetime_col (str): The name of the datetime column.
        datetime_format (str): The format of the datetime column.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    
    df_imputed = prepare_data(df_nan, datetime_col, datetime_format)
    
    df_imputed = df_imputed.interpolate(method='linear')
    if df_imputed.isnull().sum().sum() > 0:
        df_imputed = df_imputed.ffill().bfill()
    
    return df_imputed.reset_index()
