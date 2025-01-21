from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import yaml
import os


def load_dataset_etth():
    TARGET = 'OT'
    DATETIME = 'date'
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    df = pd.read_csv('data/ETTh1.csv')
    df[DATETIME] = pd.to_datetime(df[DATETIME], format=DATETIME_FORMAT)
    
    return df, DATETIME, DATETIME_FORMAT, TARGET

def add_future_rows(df: pd.DataFrame, datetime_col: str, n_rows: int) -> pd.DataFrame:
    """
    Appends n_rows to df by incrementing the datetime_col by freq
    and fills the rest of the columns with NaN.
    
    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe.
    datetime_col : str
        The name of the datetime column in df.
    n_rows : int
        Number of rows to add.
    
    Returns
    -------
    pd.DataFrame
        The original dataframe with n_rows extra rows.
    """
    try:
        freq = df[datetime_col].diff().mode()[0]
    except IndexError:
        raise ValueError("The datetime column has irregular frequencies.")
    
    # 1. Get the last datetime in the dataframe
    last_date = df[datetime_col].iloc[-1]
    
    # 2. Generate new datetimes
    new_datetimes = [last_date + (i + 1) * freq for i in range(n_rows)]
    
    # 3. Create a new DataFrame with these datetimes
    new_rows = pd.DataFrame({datetime_col: new_datetimes})
    
    # 4. For every other column, set the values to NaN
    for col in df.columns:
        if col != datetime_col:
            new_rows[col] = np.nan
            
    # 5. Concatenate to the original df
    df_extended = pd.concat([df, new_rows], ignore_index=True)
    
    return df_extended

def introduce_nans_randomly(df: pd.DataFrame, nan_percentage: float, datetime_col: str, target_column: str):
    df_nan = df.copy()
    cols = [col for col in df_nan.columns if col not in [target_column, datetime_col]]
    nan_positions = []

    # Calculate total values to be converted to NaN
    total_values = df_nan[cols].size
    n_nans = int(total_values * nan_percentage)

    # Introduce NaNs in random locations within the specified columns only
    for _ in range(n_nans):
        row_idx = np.random.randint(0, df_nan.shape[0])
        col = np.random.choice(cols)
        datetime_value = df_nan.iloc[row_idx][datetime_col]

        # Check if the cell is already NaN to avoid counting duplicates
        if pd.isnull(df_nan.iloc[row_idx, df_nan.columns.get_loc(col)]):
            continue
        df_nan.iloc[row_idx, df_nan.columns.get_loc(col)] = np.nan
        nan_positions.append((datetime_value, col))

    return df_nan, nan_positions

def introduce_nans_sequentially(df, nan_percentage, num_batches, datetime_col: str, target_column: str):
    df_copy = df.copy()
    n_rows = len(df_copy)
    eligible_columns = df_copy.columns.difference([target_column, datetime_col])

    total_cells_to_nan = int(n_rows * len(eligible_columns) * nan_percentage)
    cells_per_batch = total_cells_to_nan // num_batches

    margin = int(0.2 * n_rows)
    available_rows = n_rows - 2 * margin
    row_step = available_rows // num_batches

    nan_indices = []

    for i in range(num_batches):
        start_row = margin + i * row_step
        end_row = min(start_row + (cells_per_batch // len(eligible_columns)), n_rows)

        for row in range(start_row, end_row):
            for col in eligible_columns:
                if row < n_rows:
                    datetime_value = df_copy.iloc[row][datetime_col]
                    df_copy.iloc[row, df_copy.columns.get_loc(col)] = np.nan
                    nan_indices.append((datetime_value, col))

    return df_copy, nan_indices

def evaluate_time_series_regression(
    df: pd.DataFrame,
    datetime_col: str,
    datetime_format: str,
    target_column: str,
    models: list,
    verbose: bool = True
) -> None:
    """
    Trains multiple regression models on the given DataFrame, evaluates them using TimeSeriesSplit,
    and prints averaged evaluation metrics across all models.

    Parameters:
    df (pd.DataFrame): The input DataFrame with features and target.
    datetime_col (str): The name of the datetime column.
    datetime_format (str): The format of the datetime column.
    target_column (str): The name of the target column for regression.
    models (list): A list of regression models to be trained.
    """
    
    df = df.copy()
    
    if datetime_col in df.columns:
        if datetime_format is not None:
            df[datetime_col] = pd.to_datetime(df[datetime_col], format=datetime_format)
        else:
            df_nan[datetime_col] = pd.to_datetime(df_nan[datetime_col])
        
        df.set_index(datetime_col, inplace=True)
    
    
    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # TimeSeriesSplit cross-validator
    tscv = TimeSeriesSplit(n_splits=10)

    all_r2_scores = []
    all_mae_scores = []
    all_rmse_scores = []

    # Loop through all models
    for model in models:
        r2_scores = []
        mae_scores = []
        rmse_scores = []

        # TimeSeriesSplit cross-validation
        for train_index, test_index in tscv.split(X):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

            # Train the model
            model.fit(X_train_fold, y_train_fold)

            # Predict and calculate metrics
            y_pred = model.predict(X_test_fold)
            r2_scores.append(r2_score(y_test_fold, y_pred))
            mae_scores.append(mean_absolute_error(y_test_fold, y_pred))
            rmse_scores.append(root_mean_squared_error(y_test_fold, y_pred))

        # Aggregate metrics for this model
        if r2_scores:
            all_r2_scores.extend(r2_scores)
            all_mae_scores.extend(mae_scores)
            all_rmse_scores.extend(rmse_scores)

    if verbose:
        # Create a DataFrame to hold the metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Mean R2 Score', 'Mean Absolute Error', 'Root Mean Squared Error'],
            'Values': [np.mean(all_r2_scores), np.mean(all_mae_scores), np.mean(all_rmse_scores)]
        })  
        metrics_df.set_index('Metric', inplace=True)
        print(metrics_df)
        
    return {
            'r2': float(np.mean(all_r2_scores)),
            'mae': float(np.mean(all_mae_scores)),
            'rmse': float(np.mean(all_rmse_scores))
        }

def plot_features(
    df, 
    datetime_col, 
    datetime_format=None, 
    TARGET=None,
    start_prediction_date=None,
    save=False,
    output_file_path="plot.png", 
    title="Feature Plots",
    dpi=500
):
    df = df.copy()

    # Convert to DateTimeIndex
    if datetime_col in df.columns:
        df[datetime_col] = (pd.to_datetime(df[datetime_col], format=datetime_format)
                            if datetime_format else pd.to_datetime(df[datetime_col]))
        df.set_index(datetime_col, inplace=True)

    # Exclude the target from columns_to_plot
    columns_to_plot = [col for col in df.columns if col != TARGET]
    num_features = len(columns_to_plot)

    rows = 2
    cols = math.ceil(num_features / rows)
    leftover = rows * cols - num_features

    for r in range(2, num_features + 1):
        c = math.ceil(num_features / r)
        if c == 1:
            continue
        l = r * c - num_features
        if l < leftover or (l == leftover and r < rows):
            rows, cols, leftover = r, c, l

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), dpi=dpi)
    fig.suptitle(title, fontsize=25)

    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()

    if start_prediction_date:
        
        # draw a vertical line at the start_prediction_date
        start_pred_dt = pd.to_datetime(start_prediction_date, format=datetime_format
                                        if datetime_format else None)
        
    else:
        start_pred_dt = None

    for i, column in enumerate(columns_to_plot):
        axes[i].plot(df.index, df[column], label=column)
        axes[i].set_title(f"Feature: {column}")
        axes[i].set_xlabel("Datetime")
        axes[i].set_ylabel(column)
        axes[i].grid(True)

        # Add the vertical line if parsed date is valid
        if start_pred_dt is not None and not pd.isna(start_pred_dt):
            axes[i].axvline(start_pred_dt, color='red', linestyle='--', label='Start Prediction')

        # Rotate x-ticks, then place legend
        for tick in axes[i].get_xticklabels():
            tick.set_rotation(45)
        axes[i].legend()

    # Remove leftover axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        if os.path.dirname(output_file_path):
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        plt.savefig(output_file_path, dpi=dpi)
        print(f"Plot saved to {output_file_path}")

    plt.show()

def save_results(results, runtimes, name, folder_name="results"):
    """
    Create a folder in the current directory and save the provided dictionary as a YAML file.

    Parameters:
    - results (dict): The dictionary to be saved in the YAML file.
    - folder_name (str): The name of the folder to be created. Default is "results_folder".
    """    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    folder_name = os.path.join(folder_name, name)
    os.makedirs(folder_name, exist_ok=True)    
    
    datasets = results["datasets"]
    results = results["values"]
    
    # Write the filtered dictionary to the YAML file
    with open(os.path.join(folder_name, "results.yaml"), 'w') as yaml_file:
        yaml.dump(results, yaml_file, default_flow_style=False)
        
    # Write the runtimes to the YAML file
    with open(os.path.join(folder_name, "runtimes.yaml"), 'w') as yaml_file:
        yaml.dump(runtimes, yaml_file, default_flow_style=False)
        
    # Now create an img folder inside the folder
    os.makedirs(os.path.join(folder_name, 'imgs'), exist_ok=True)
    
    # Plot the results and save the images
    plot_results(results, runtimes, os.path.join(folder_name, 'imgs'))
    
    # Saving the datasets
    datasets_folder = os.path.join(folder_name, 'datasets')
    os.makedirs(datasets_folder, exist_ok=True)
    
    original_folder = os.path.join(datasets_folder, 'original')    
    os.makedirs(original_folder, exist_ok=True)
    og_df, DATETIME, DATETIME_FORMAT, name = datasets["original"]

    # Save the original dataset
    og_df.to_csv(os.path.join(original_folder, 'dataset.csv'), index=False)
    plot_features(og_df, DATETIME, DATETIME_FORMAT, save=True, output_file_path=os.path.join(original_folder, 'plot.png'), title=f"Imputed Dataset({name}): Original")

    # Save the imputed datasets based on the new structure
    for mode, methods in datasets["imputed"].items():
        mode_folder = os.path.join(datasets_folder, mode)
        os.makedirs(mode_folder, exist_ok=True)
        for method, d in methods.items(): 
            temp_df, DATETIME, DATETIME_FORMAT, name = d
            method_folder = os.path.join(mode_folder, method)
            os.makedirs(method_folder, exist_ok=True)
            plot_features(temp_df, DATETIME, DATETIME_FORMAT, save=True, output_file_path=os.path.join(method_folder, f'{name}-{mode}-{method}-plot.png'), title=f"Imputed Dataset({name}):({mode})-({method})")
            temp_df.to_csv(os.path.join(method_folder, f'{name}-{mode}-{method}-dataset.csv'), index=False)

    print(f"Results saved in {folder_name}")

def evaluate_imputation(data, imputed_data, nan_positions, datetime, datetime_format, verbose=False):
    # Get actual and imputed values by column
    values_by_column = {}
    for timestamp, column in nan_positions:
        # Locate the row in the original and imputed data
        real_value = data.loc[data[datetime] == timestamp.strftime(datetime_format), column].values[0]
        imputed_value = imputed_data.loc[imputed_data[datetime] == timestamp.strftime(datetime_format), column].values[0]

        # Append to the respective column in the dictionary
        if column not in values_by_column:
            values_by_column[column] = []
        values_by_column[column].append((real_value, imputed_value))

    # Initialize lists to store NMAPE, NRMSE, and ACF arrays for each column
    nmae_list = []
    nrmse_list = []
    acf_original_list = []
    acf_imputed_list = []

    for column, values in values_by_column.items():
        # Unpack actual and predicted values
        actuals, predictions = zip(*values)

        # Convert to numpy arrays for easier calculations
        actuals = np.array(actuals)
        predictions = np.array(predictions)

        # Calculate the range of the column
        column_range = np.max(actuals) - np.min(actuals)

        # Avoid division by zero
        if column_range == 0:
            continue

        # Calculate NMAPE
        nmae = np.mean(np.abs(actuals - predictions)) / column_range
        nmae_list.append(nmae)

        # Calculate NRMSE
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        nrmse = rmse / column_range
        nrmse_list.append(nrmse)

        # Calculate ACF Error and accumulate arrays
        acf_original = acf(actuals, nlags=15)
        acf_imputed = acf(predictions, nlags=15)
        acf_original_list.append(acf_original)
        acf_imputed_list.append(acf_imputed)

    # Calculate the mean ACF arrays
    mean_acf_original = np.mean(np.vstack(acf_original_list), axis=0)
    mean_acf_imputed = np.mean(np.vstack(acf_imputed_list), axis=0)

    # Prepare the scores dictionary
    scores = {
        'nmae': float(np.mean(nmae_list)),
        'nrmse': float(np.mean(nrmse_list)),
        'ac_score_original': mean_acf_original,
        'ac_score_imputed': mean_acf_imputed
    }

    # Print the metrics if verbose is True
    if verbose:
        metrics_df = pd.DataFrame({
            'Metric': ['Mean NMAPE', 'Mean NRMSE', 'Mean ACF Error'],
            'Values': [scores["nmae"], scores["nrmse"], np.mean(np.abs(mean_acf_original - mean_acf_imputed))]
        })
        print(metrics_df.set_index('Metric'))

    # Return calculated metrics
    return scores

def plot_results(results, runtimes, output_dir="results/imgs"):
    """
    Plots four graphs (regression_errors, regression_r2_scores, imputation_errors, imputation_r2_scores)
    from the provided results dictionary and saves them as PNG files in the specified output directory.

    Args:
    - results (dict): Dictionary containing all the data to plot.
    - runtimes (dict): Dictionary containing execution times for different methods.
    - output_dir (str): Directory to save the output PNG files.
    """
    def save_and_show_plot(output_file):
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()
        
    fontsize = 13

    # Regression Errors
    regression_errors = results["regression_errors"]
    approaches = list(regression_errors.keys())

    mae_values, rmse_values, x_positions, x_labels = [], [], [], []
    current_x, approach_positions = 0, []

    for approach in approaches:
        start_position = current_x
        if approach == "original":
            mae_values.append(regression_errors[approach]["mae"])
            rmse_values.append(regression_errors[approach]["rmse"])
            x_positions.append(current_x)
            x_labels.append("Original")
            current_x += 1.5
        else:
            methods = regression_errors[approach].keys()
            for method in methods:
                mae_values.append(regression_errors[approach][method]["mae"])
                rmse_values.append(regression_errors[approach][method]["rmse"])
                x_positions.append(current_x)
                x_labels.append(method.capitalize())
                current_x += 0.75
        end_position = current_x
        approach_positions.append((start_position + end_position - 0.75) / 2)
        current_x += 1.5

    bar_width = 0.35
    plt.figure(figsize=(14, 8))
    plt.bar(np.array(x_positions) - bar_width / 2, mae_values, width=bar_width, label="MAE", alpha=0.8)
    plt.bar(np.array(x_positions) + bar_width / 2, rmse_values, width=bar_width, label="RMSE", alpha=0.8)
    plt.xlabel("Methods", fontsize=fontsize, labelpad=20)
    plt.ylabel("Errors", fontsize=fontsize)
    plt.title("Regression Errors Across Approaches and Methods", fontsize=14)
    plt.xticks(x_positions, x_labels, fontsize=10, rotation=45, ha="right")
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    approach_labels = [approach.replace("_", " ").capitalize() for approach in approaches]
    ax_secondary = plt.gca().secondary_xaxis("top")
    ax_secondary.set_xticks(approach_positions)
    ax_secondary.set_xticklabels(approach_labels, fontsize=fontsize)
    ax_secondary.set_xlabel("Approaches", fontsize=fontsize, labelpad=10)

    save_and_show_plot(f"{output_dir}/regression_errors.png")

    # Regression R2 Scores
    regression_r2_scores = results["regression_r2_score"]
    r2_values, x_positions, x_labels = [], [], []
    current_x, approach_positions = 0, []

    for approach in approaches:
        start_position = current_x
        if approach == "original":
            r2_values.append(regression_r2_scores[approach])
            x_positions.append(current_x)
            x_labels.append("Original")
            current_x += 1.5
        else:
            methods = regression_r2_scores[approach].keys()
            for method in methods:
                r2_values.append(regression_r2_scores[approach][method])
                x_positions.append(current_x)
                x_labels.append(method.capitalize())
                current_x += 0.75
        end_position = current_x
        approach_positions.append((start_position + end_position - 0.75) / 2)
        current_x += 1.5

    plt.figure(figsize=(14, 8))
    plt.bar(x_positions, r2_values, width=bar_width, alpha=0.8, color="salmon", label="R2 Scores")
    plt.xlabel("Methods", fontsize=fontsize, labelpad=20)
    plt.ylabel("R2 Scores", fontsize=fontsize)
    plt.title("Regression R2 Scores Across Approaches and Methods", fontsize=14)
    plt.xticks(x_positions, x_labels, fontsize=10, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    ax_secondary = plt.gca().secondary_xaxis("top")
    ax_secondary.set_xticks(approach_positions)
    ax_secondary.set_xticklabels(approach_labels, fontsize=fontsize)
    ax_secondary.set_xlabel("Approaches", fontsize=fontsize, labelpad=10)

    save_and_show_plot(f"{output_dir}/regression_r2_score.png")

    # Imputation Errors
    imputation_errors = {k: v for k, v in results["imputation_errors"].items() if k != "ac_score_original"}
    approaches = list(imputation_errors.keys())
    methods = next(iter(imputation_errors.values())).keys()

    mape_values, rmse_values, x_positions, x_labels = [], [], [], []
    current_x, approach_positions = 0, []

    for approach in approaches:
        start_position = current_x
        for method in methods:
            mape_values.append(imputation_errors[approach][method]["nmae"])
            rmse_values.append(imputation_errors[approach][method]["nrmse"])
            x_positions.append(current_x)
            x_labels.append(method.capitalize())
            current_x += 0.75
        end_position = current_x
        approach_positions.append((start_position + end_position - 0.75) / 2)
        current_x += 1.5

    plt.figure(figsize=(14, 8))
    plt.bar(np.array(x_positions) - bar_width / 2, mape_values, width=bar_width, label="N-MAE", alpha=0.8)
    plt.bar(np.array(x_positions) + bar_width / 2, rmse_values, width=bar_width, label="N-RMSE", alpha=0.8)
    plt.xlabel("Methods", fontsize=fontsize, labelpad=20)
    plt.ylabel("Mean Errors of Features", fontsize=fontsize)
    plt.title("Mean Imputation Errors of Features Across Approaches and Methods", fontsize=14)
    plt.xticks(x_positions, x_labels, fontsize=10, rotation=45, ha="right")
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    ax_secondary = plt.gca().secondary_xaxis("top")
    ax_secondary.set_xticks(approach_positions)
    ax_secondary.set_xticklabels([approach.replace("_", " ").capitalize() for approach in approaches], fontsize=fontsize)
    ax_secondary.set_xlabel("Approaches", fontsize=fontsize, labelpad=10)

    save_and_show_plot(f"{output_dir}/imputation_errors.png")
    
    # AC Error Plot
    ac_scores = results["imputation_errors"]
    ac_original = ac_scores["ac_score_original"]
    approaches = [key for key in ac_scores.keys() if key != "ac_score_original"]
    methods = list(ac_scores["20% of data removed in random rows"].keys())

    fig, axes = plt.subplots(1, len(approaches), figsize=(18, 6), sharey=True)

    for i, approach in enumerate(approaches):
        ax = axes[i]
        for method in methods:
            ac_values = ac_scores[approach][method]["ac_score_imputed"]
            ax.plot(ac_values, marker="o", label=method.capitalize())

        ax.plot(ac_original[approach], linestyle="--", color="black", label="Original")

        ax.set_title(approach.replace("_", " ").capitalize())
        ax.set_xlabel("Lag")
        if i == 0:
            ax.set_ylabel("ACF")
        ax.legend()
    save_and_show_plot(f"{output_dir}/ac_graph.png")


    # Runtimes
    methods = runtimes.keys()
    times = [runtimes[method] for method in methods]
    x_positions = list(range(len(methods)))

    plt.figure(figsize=(10, 6))
    plt.bar(x_positions, times, color="limegreen")
    plt.xlabel("Methods", fontsize=fontsize, labelpad=20)
    plt.ylabel("Time (seconds)", fontsize=fontsize)
    plt.title("Execution Time for Different Methods", fontsize=14)
    plt.xticks(x_positions, [method.capitalize() for method in methods], fontsize=10, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    save_and_show_plot(f"{output_dir}/execution_time.png")

