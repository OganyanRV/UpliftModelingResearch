from src.global_params import *

from itertools import accumulate 
from sklearn.utils.validation import check_consistent_length
from sklift.metrics import (
    uplift_curve, perfect_uplift_curve, uplift_auc_score,
    qini_curve, perfect_qini_curve, qini_auc_score,
    treatment_balance_curve, uplift_by_percentile
)
import causalml.metrics as cmetrics
import numpy as np
import pandas as pd


# from sklift.metrics import uplift_curve -- same ig
def calculate_plot_for_uplift_curve(df):
    population = []
    uplift = []
    
    values = list(df[["score", COL_TREATMENT, COL_TARGET]].values)
    values.sort(key=lambda x: -x[0])
    
    num_treatment = 0
    num_control = 0
    sum_target_treatment = 0
    sum_target_control = 0
    
    for cur_num_population, value in enumerate(values):
        population.append(cur_num_population)
        
        if value[1] == 1:
            num_treatment+=1
            if value[2] == 1:
                sum_target_treatment +=1
                
        if value[1] == 0:
            num_control+=1
            if value[2] == 1:
                sum_target_control +=1
        
        cur_uplift = (sum_target_treatment/max(num_treatment, 1) - sum_target_control/max(num_control, 1)) * (cur_num_population + 1)
        uplift.append(cur_uplift)

    return population, uplift

def plot_aucs(*args, labels, ds_name):
    num_models = []
    uplift_models = []

    for predicted in args:
        num_model, uplift_model  = calculate_plot_for_uplift_curve(predicted)
        num_models.append(num_model)
        uplift_models.append(uplift_model)
    plt.figure(figsize=(8, 6))
    for i in range(len(uplift_models)):
        plt.plot(num_models[i], uplift_models[i], label=labels[i])
    plt.xlabel('Population')
    plt.ylabel('Uplift')
    plt.title(f'Comparison of Uplift Curves on {ds_name}')
    plt.legend()
        
    
def get_auuc(predicted):
    """
    Возвращает AUUC модели
    """
    # ml_auuc, random_auuc = cmetrics.auuc_score(
    #     predicted, 
    #     outcome_col=COL_TARGET, 
    #     treatment_col=COL_TREATMENT, 
    # )

    return uplift_auc_score(predicted[COL_TARGET], predicted.score, predicted[COL_TREATMENT])

def get_qini(predicted):
    """
    Возвращает AUUQ модели
    """
    return qini_auc_score(predicted[COL_TARGET], predicted.score, predicted[COL_TREATMENT])

# Переписывал библиотечные функции scikit-uplift так как не было функционала CUMMULATIVE    
# 2 переписал под свои нужды, остальные скопировал и поправил пару строк, в которых либа багует
def uplift_by_percentile_CUM(y_true, uplift, treatment, strategy='overall',
                         bins=10, std=False, total=False, string_percentiles=True):
    """
    Считает кумулятивный uplift@percentile
    """
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    strategy_methods = ['overall', 'by_group']

    n_samples = len(y_true)

    if strategy not in strategy_methods:
        raise ValueError(f'Response rate supports only calculating methods in {strategy_methods},'
                         f' got {strategy}.')

    if not isinstance(total, bool):
        raise ValueError(f'Flag total should be bool: True or False.'
                         f' Invalid value total: {total}')

    if not isinstance(std, bool):
        raise ValueError(f'Flag std should be bool: True or False.'
                         f' Invalid value std: {std}')

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f'Bins should be positive integer.'
                         f' Invalid value bins: {bins}')

    if bins >= n_samples:
        raise ValueError(f'Number of bins = {bins} should be smaller than the length of y_true {n_samples}')

    if not isinstance(string_percentiles, bool):
        raise ValueError(f'string_percentiles flag should be bool: True or False.'
                         f' Invalid value string_percentiles: {string_percentiles}')

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    response_rate_trmnt, variance_trmnt, n_trmnt = response_rate_by_percentile(
        y_true, uplift, treatment, group='treatment', strategy=strategy, bins=bins)

    response_rate_ctrl, variance_ctrl, n_ctrl = response_rate_by_percentile(
        y_true, uplift, treatment, group='control', strategy=strategy, bins=bins)

    good_trmnt = np.array(n_trmnt * response_rate_trmnt, dtype=np.int64)
    good_ctrl = np.array(n_ctrl * response_rate_ctrl, dtype=np.int64)

    good_trmnt = np.array(list(accumulate(good_trmnt)))
    good_ctrl = np.array(list(accumulate(good_ctrl)))

    n_trmnt = np.array(list(accumulate(n_trmnt)))
    n_ctrl = np.array(list(accumulate(n_ctrl)))

    response_rate_ctrl = good_ctrl / n_ctrl
    response_rate_trmnt = good_trmnt / n_trmnt
    
    uplift_scores = response_rate_trmnt - response_rate_ctrl
    uplift_variance = variance_trmnt + variance_ctrl

    percentiles = [round(p * 100 / bins) for p in range(1, bins + 1)]

    if string_percentiles:
        percentiles = [f"{percentiles[i]}" for i in range(len(percentiles))]

    df = pd.DataFrame({
        'percentile': percentiles,
        'n_treatment': n_trmnt,
        'n_control': n_ctrl,
        'response_rate_treatment': response_rate_trmnt,
        'response_rate_control': response_rate_ctrl,
        'uplift': uplift_scores
    })
    

    if total:
        response_rate_trmnt_total, variance_trmnt_total, n_trmnt_total = response_rate_by_percentile(
            y_true, uplift, treatment, strategy=strategy, group='treatment', bins=1)

        response_rate_ctrl_total, variance_ctrl_total, n_ctrl_total = response_rate_by_percentile(
            y_true, uplift, treatment, strategy=strategy, group='control', bins=1)

        df.loc[-1, :] = ['total', n_trmnt_total[0], n_ctrl_total[0], response_rate_trmnt_total[0],
                         response_rate_ctrl_total[0], response_rate_trmnt_total[0] - response_rate_ctrl_total[0]]

    if std:
        std_treatment = np.sqrt(variance_trmnt)
        std_control = np.sqrt(variance_ctrl)
        std_uplift = np.sqrt(uplift_variance)

        if total:
            std_treatment = np.append(std_treatment, np.sum(std_treatment))
            std_control = np.append(std_control, np.sum(std_control))
            std_uplift = np.append(std_uplift, np.sum(std_uplift))

        df.loc[:, 'std_treatment'] = std_treatment
        df.loc[:, 'std_control'] = std_control
        df.loc[:, 'std_uplift'] = std_uplift

    df = df \
        .set_index('percentile', drop=True, inplace=False) \
        .astype({'n_treatment': 'int32', 'n_control': 'int32'})

    return df


def plot_uplift_by_percentile_CUM(y_true, uplift, treatment, strategy='overall',
                              kind='line', bins=10, string_percentiles=True):
    """Plot uplift score, treatment response rate and control response rate at each percentile.

    Treatment response rate ia a target mean in the treatment group.
    Control response rate is a target mean in the control group.
    Uplift score is a difference between treatment response rate and control response rate.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        strategy (string, ['overall', 'by_group']): Determines the calculating strategy. Default is 'overall'.

            * ``'overall'``:
                The first step is taking the first k observations of all test data ordered by uplift prediction
                (overall both groups - control and treatment) and conversions in treatment and control groups
                calculated only on them. Then the difference between these conversions is calculated.
            * ``'by_group'``:
                Separately calculates conversions in top k observations in each group (control and treatment)
                sorted by uplift predictions. Then the difference between these conversions is calculated.

        kind (string, ['line', 'bar']): The type of plot to draw. Default is 'line'.

            * ``'line'``:
                Generates a line plot.
            * ``'bar'``:
                Generates a traditional bar-style plot.

        bins (int): Determines а number of bins (and the relative percentile) in the test data. Default is 10.
        string_percentiles (bool): type of xticks: float or string to plot. Default is True (string).

    Returns:
        Object that stores computed values.
    """

    strategy_methods = ['overall', 'by_group']
    kind_methods = ['line', 'bar']
    
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    n_samples = len(y_true)

    if strategy not in strategy_methods:
        raise ValueError(f'Response rate supports only calculating methods in {strategy_methods},'
                         f' got {strategy}.')

    if kind not in kind_methods:
        raise ValueError(f'Function supports only types of plots in {kind_methods},'
                         f' got {kind}.')

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(
            f'Bins should be positive integer. Invalid value bins: {bins}')

    if bins >= n_samples:
        raise ValueError(
            f'Number of bins = {bins} should be smaller than the length of y_true {n_samples}')

    if not isinstance(string_percentiles, bool):
        raise ValueError(f'string_percentiles flag should be bool: True or False.'
                         f' Invalid value string_percentiles: {string_percentiles}')
    df = uplift_by_percentile_CUM(y_true, uplift, treatment, strategy=strategy,
                              std=True, total=True, bins=bins, string_percentiles=False)

    percentiles = df.index[:bins].values.astype(float)

    response_rate_trmnt = df.loc[percentiles, 'response_rate_treatment'].values
    std_trmnt = df.loc[percentiles, 'std_treatment'].values

    response_rate_ctrl = df.loc[percentiles, 'response_rate_control'].values
    std_ctrl = df.loc[percentiles, 'std_control'].values

    uplift_score = df.loc[percentiles, 'uplift'].values
    std_uplift = df.loc[percentiles, 'std_uplift'].values

    uplift_weighted_avg = df.loc['total', 'uplift']
    check_consistent_length(percentiles, response_rate_trmnt,
                            response_rate_ctrl, uplift_score,
                            std_trmnt, std_ctrl, std_uplift)

    
    if kind == 'line':
        _, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        axes.errorbar(percentiles, response_rate_trmnt, yerr=std_trmnt,
                      linewidth=2, color='forestgreen', label='treatment\nresponse rate')
        axes.errorbar(percentiles, response_rate_ctrl, yerr=std_ctrl,
                      linewidth=2, color='orange', label='control\nresponse rate')
        axes.errorbar(percentiles, uplift_score, yerr=std_uplift,
                      linewidth=2, color='red', label='uplift')
        axes.fill_between(percentiles, response_rate_trmnt,
                          response_rate_ctrl, alpha=0.1, color='red')

        if np.amin(uplift_score) < 0:
            axes.axhline(y=0, color='black', linewidth=1)

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + \
                              [f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}" for i in range(len(percentiles) - 1)]
            axes.set_xticks(percentiles)
            axes.set_xticklabels(percentiles_str, rotation=45)
        else:
            axes.set_xticks(percentiles)

        axes.legend(loc='upper right')
        axes.set_title(
            f'Uplift by percentile\nweighted average uplift = {uplift_weighted_avg:.4f}')
        axes.set_xlabel('Percentile')
        axes.set_ylabel(
            'Uplift = treatment response rate - control response rate')

    else:  # kind == 'bar'
        delta = percentiles[0]
        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(13, 9), sharex=True, sharey=True)
        fig.text(0.01, 0.5, 'Uplift = treatment response rate - control response rate',
                 va='center', ha='center', rotation='vertical')

        axes[1].bar(np.array(percentiles) - delta / 6, response_rate_trmnt, delta / 3,
                    yerr=std_trmnt, color='forestgreen', label='treatment\nresponse rate')
        axes[1].bar(np.array(percentiles) + delta / 6, response_rate_ctrl, delta / 3,
                    yerr=std_ctrl, color='orange', label='control\nresponse rate')
        axes[0].bar(np.array(percentiles), uplift_score, delta / 1.5,
                    yerr=std_uplift, color='red', label='uplift')

        axes[0].legend(loc='upper right')
        axes[0].tick_params(axis='x', bottom=False)
        axes[0].axhline(y=0, color='black', linewidth=1)
        axes[0].set_title(
            f'Uplift by percentile\nweighted average uplift = {uplift_weighted_avg:.4f}')

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + \
                              [f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}" for i in range(len(percentiles) - 1)]
            axes[1].set_xticks(percentiles)
            axes[1].set_xticklabels(percentiles_str, rotation=45)

        else:
            axes[1].set_xticks(percentiles)

        axes[1].legend(loc='upper right')
        axes[1].axhline(y=0, color='black', linewidth=1)
        axes[1].set_xlabel('Percentile')
        axes[1].set_title('Response rate by percentile')

    return axes

def response_rate_by_percentile(y_true, uplift, treatment, group, strategy='overall', bins=10):
    """Compute response rate (target mean in the control or treatment group) at each percentile.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        group (string, ['treatment', 'control']): Group type for computing response rate: treatment or control.

            * ``'treatment'``:
                Values equal 1 in the treatment column.
            * ``'control'``:
                Values equal 0 in the treatment column.

        strategy (string, ['overall', 'by_group']): Determines the calculating strategy. Default is 'overall'.

            * ``'overall'``:
                The first step is taking the first k observations of all test data ordered by uplift prediction
                (overall both groups - control and treatment) and conversions in treatment and control groups
                calculated only on them. Then the difference between these conversions is calculated.
            * ``'by_group'``:
                Separately calculates conversions in top k observations in each group (control and treatment)
                sorted by uplift predictions. Then the difference between these conversions is calculated.

        bins (int): Determines the number of bins (and relative percentile) in the data. Default is 10.
        
    Returns:
        array (shape = [>2]), array (shape = [>2]), array (shape = [>2]):
        response rate at each percentile for control or treatment group,
        variance of the response rate at each percentile,
        group size at each percentile.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    group_types = ['treatment', 'control']
    strategy_methods = ['overall', 'by_group']
    
    n_samples = len(y_true)
    
    if group not in group_types:
        raise ValueError(f'Response rate supports only group types in {group_types},'
                         f' got {group}.') 

    if strategy not in strategy_methods:
        raise ValueError(f'Response rate supports only calculating methods in {strategy_methods},'
                         f' got {strategy}.')
    
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f'Bins should be positive integer. Invalid value bins: {bins}')

    if bins >= n_samples:
        raise ValueError(f'Number of bins = {bins} should be smaller than the length of y_true {n_samples}')
    
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    order = np.argsort(uplift, kind='mergesort')[::-1]

    trmnt_flag = 1 if group == 'treatment' else 0
    
    if strategy == 'overall':
        y_true_bin = np.array_split(y_true[order], bins)
        trmnt_bin = np.array_split(treatment[order], bins)
        
        group_size = np.array([len(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)])
        response_rate = np.array([np.mean(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)])

    else:  # strategy == 'by_group'
        y_bin = np.array_split(y_true[order][treatment[order] == trmnt_flag], bins)
        
        group_size = np.array([len(y) for y in y_bin])
        response_rate = np.array([np.mean(y) for y in y_bin])

    variance = np.multiply(response_rate, np.divide((1 - response_rate), group_size))

    return response_rate, variance, group_size
def uplift_by_percentile(y_true, uplift, treatment, strategy='overall',
                         bins=10, std=False, total=False, string_percentiles=True):
    """Compute metrics: uplift, group size, group response rate, standard deviation at each percentile.

    Metrics in columns and percentiles in rows of pandas DataFrame:

        - ``n_treatment``, ``n_control`` - group sizes.
        - ``response_rate_treatment``, ``response_rate_control`` - group response rates.
        - ``uplift`` - treatment response rate substract control response rate.
        - ``std_treatment``, ``std_control`` - (optional) response rates standard deviation.
        - ``std_uplift`` - (optional) uplift standard deviation.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        strategy (string, ['overall', 'by_group']): Determines the calculating strategy. Default is 'overall'.

            * ``'overall'``:
                The first step is taking the first k observations of all test data ordered by uplift prediction
                (overall both groups - control and treatment) and conversions in treatment and control groups
                calculated only on them. Then the difference between these conversions is calculated.
            * ``'by_group'``:
                Separately calculates conversions in top k observations in each group (control and treatment)
                sorted by uplift predictions. Then the difference between these conversions is calculated

        std (bool): If True, add columns with the uplift standard deviation and the response rate standard deviation.
            Default is False.
        total (bool): If True, add the last row with the total values. Default is False.
            The total uplift computes as a total response rate treatment - a total response rate control.
            The total response rate is a response rate on the full data amount.
        bins (int): Determines the number of bins (and the relative percentile) in the data. Default is 10.
        string_percentiles (bool): type of percentiles in the index: float or string. Default is True (string).

    Returns:
        pandas.DataFrame: DataFrame where metrics are by columns and percentiles are by rows.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    strategy_methods = ['overall', 'by_group']

    n_samples = len(y_true)

    if strategy not in strategy_methods:
        raise ValueError(f'Response rate supports only calculating methods in {strategy_methods},'
                         f' got {strategy}.')

    if not isinstance(total, bool):
        raise ValueError(f'Flag total should be bool: True or False.'
                         f' Invalid value total: {total}')

    if not isinstance(std, bool):
        raise ValueError(f'Flag std should be bool: True or False.'
                         f' Invalid value std: {std}')

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f'Bins should be positive integer.'
                         f' Invalid value bins: {bins}')

    if bins >= n_samples:
        raise ValueError(f'Number of bins = {bins} should be smaller than the length of y_true {n_samples}')

    if not isinstance(string_percentiles, bool):
        raise ValueError(f'string_percentiles flag should be bool: True or False.'
                         f' Invalid value string_percentiles: {string_percentiles}')

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    response_rate_trmnt, variance_trmnt, n_trmnt = response_rate_by_percentile(
        y_true, uplift, treatment, group='treatment', strategy=strategy, bins=bins)

    response_rate_ctrl, variance_ctrl, n_ctrl = response_rate_by_percentile(
        y_true, uplift, treatment, group='control', strategy=strategy, bins=bins)

    uplift_scores = response_rate_trmnt - response_rate_ctrl
    uplift_variance = variance_trmnt + variance_ctrl

    percentiles = [round(p * 100 / bins) for p in range(1, bins + 1)]

    if string_percentiles:
        percentiles = [f"0-{percentiles[0]}"] + \
            [f"{percentiles[i]}-{percentiles[i + 1]}" for i in range(len(percentiles) - 1)]


    df = pd.DataFrame({
        'percentile': percentiles,
        'n_treatment': n_trmnt,
        'n_control': n_ctrl,
        'response_rate_treatment': response_rate_trmnt,
        'response_rate_control': response_rate_ctrl,
        'uplift': uplift_scores
    })
    

    if total:
        response_rate_trmnt_total, variance_trmnt_total, n_trmnt_total = response_rate_by_percentile(
            y_true, uplift, treatment, strategy=strategy, group='treatment', bins=1)

        response_rate_ctrl_total, variance_ctrl_total, n_ctrl_total = response_rate_by_percentile(
            y_true, uplift, treatment, strategy=strategy, group='control', bins=1)

        df.loc[-1, :] = ['total', n_trmnt_total[0], n_ctrl_total[0], response_rate_trmnt_total[0],
                         response_rate_ctrl_total[0], response_rate_trmnt_total[0] - response_rate_ctrl_total[0]]

    if std:
        std_treatment = np.sqrt(variance_trmnt)
        std_control = np.sqrt(variance_ctrl)
        std_uplift = np.sqrt(uplift_variance)

        if total:
            std_treatment = np.append(std_treatment, np.sum(std_treatment))
            std_control = np.append(std_control, np.sum(std_control))
            std_uplift = np.append(std_uplift, np.sum(std_uplift))

        df.loc[:, 'std_treatment'] = std_treatment
        df.loc[:, 'std_control'] = std_control
        df.loc[:, 'std_uplift'] = std_uplift

    df = df \
        .set_index('percentile', drop=True, inplace=False) \
        .astype({'n_treatment': 'int32', 'n_control': 'int32'})

    return df

def check_is_binary(array):
    """Checker if array consists of int or float binary values 0 (0.) and 1 (1.)

    Args:
        array (1d array-like): Array to check.
    """

    if not np.all(np.unique(array) == np.array([0, 1])):
        raise ValueError(f"Input array is not binary. "
                         f"Array should contain only int or float binary values 0 (or 0.) and 1 (or 1.). "
                         f"Got values {np.unique(array)}.")
def plot_uplift_by_percentile(y_true, uplift, treatment, strategy='overall',
                              kind='line', bins=10, string_percentiles=True):
    """Plot uplift score, treatment response rate and control response rate at each percentile.

    Treatment response rate ia a target mean in the treatment group.
    Control response rate is a target mean in the control group.
    Uplift score is a difference between treatment response rate and control response rate.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        strategy (string, ['overall', 'by_group']): Determines the calculating strategy. Default is 'overall'.

            * ``'overall'``:
                The first step is taking the first k observations of all test data ordered by uplift prediction
                (overall both groups - control and treatment) and conversions in treatment and control groups
                calculated only on them. Then the difference between these conversions is calculated.
            * ``'by_group'``:
                Separately calculates conversions in top k observations in each group (control and treatment)
                sorted by uplift predictions. Then the difference between these conversions is calculated.

        kind (string, ['line', 'bar']): The type of plot to draw. Default is 'line'.

            * ``'line'``:
                Generates a line plot.
            * ``'bar'``:
                Generates a traditional bar-style plot.

        bins (int): Determines а number of bins (and the relative percentile) in the test data. Default is 10.
        string_percentiles (bool): type of xticks: float or string to plot. Default is True (string).

    Returns:
        Object that stores computed values.
    """

    strategy_methods = ['overall', 'by_group']
    kind_methods = ['line', 'bar']
    
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    n_samples = len(y_true)

    if strategy not in strategy_methods:
        raise ValueError(f'Response rate supports only calculating methods in {strategy_methods},'
                         f' got {strategy}.')

    if kind not in kind_methods:
        raise ValueError(f'Function supports only types of plots in {kind_methods},'
                         f' got {kind}.')

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(
            f'Bins should be positive integer. Invalid value bins: {bins}')

    if bins >= n_samples:
        raise ValueError(
            f'Number of bins = {bins} should be smaller than the length of y_true {n_samples}')

    if not isinstance(string_percentiles, bool):
        raise ValueError(f'string_percentiles flag should be bool: True or False.'
                         f' Invalid value string_percentiles: {string_percentiles}')
    df = uplift_by_percentile(y_true, uplift, treatment, strategy=strategy,
                              std=True, total=True, bins=bins, string_percentiles=False)

    percentiles = df.index[:bins].values.astype(float)

    response_rate_trmnt = df.loc[percentiles, 'response_rate_treatment'].values
    std_trmnt = df.loc[percentiles, 'std_treatment'].values

    response_rate_ctrl = df.loc[percentiles, 'response_rate_control'].values
    std_ctrl = df.loc[percentiles, 'std_control'].values

    uplift_score = df.loc[percentiles, 'uplift'].values
    std_uplift = df.loc[percentiles, 'std_uplift'].values

    uplift_weighted_avg = df.loc['total', 'uplift']
    check_consistent_length(percentiles, response_rate_trmnt,
                            response_rate_ctrl, uplift_score,
                            std_trmnt, std_ctrl, std_uplift)

    
    if kind == 'line':
        _, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        axes.errorbar(percentiles, response_rate_trmnt, yerr=std_trmnt,
                      linewidth=2, color='forestgreen', label='treatment\nresponse rate')
        axes.errorbar(percentiles, response_rate_ctrl, yerr=std_ctrl,
                      linewidth=2, color='orange', label='control\nresponse rate')
        axes.errorbar(percentiles, uplift_score, yerr=std_uplift,
                      linewidth=2, color='red', label='uplift')
        axes.fill_between(percentiles, response_rate_trmnt,
                          response_rate_ctrl, alpha=0.1, color='red')

        if np.amin(uplift_score) < 0:
            axes.axhline(y=0, color='black', linewidth=1)

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + \
                              [f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}" for i in range(len(percentiles) - 1)]
            axes.set_xticks(percentiles)
            axes.set_xticklabels(percentiles_str, rotation=45)
        else:
            axes.set_xticks(percentiles)

        axes.legend(loc='upper right')
        axes.set_title(
            f'Uplift by percentile\nweighted average uplift = {uplift_weighted_avg:.4f}')
        axes.set_xlabel('Percentile')
        axes.set_ylabel(
            'Uplift = treatment response rate - control response rate')

    else:  # kind == 'bar'
        delta = percentiles[0]
        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(13, 9), sharex=True, sharey=True)
        fig.text(0.01, 0.5, 'Uplift = treatment response rate - control response rate',
                 va='center', ha='center', rotation='vertical')

        axes[1].bar(np.array(percentiles) - delta / 6, response_rate_trmnt, delta / 3,
                    yerr=std_trmnt, color='forestgreen', label='treatment\nresponse rate')
        axes[1].bar(np.array(percentiles) + delta / 6, response_rate_ctrl, delta / 3,
                    yerr=std_ctrl, color='orange', label='control\nresponse rate')
        axes[0].bar(np.array(percentiles), uplift_score, delta / 1.5,
                    yerr=std_uplift, color='red', label='uplift')

        axes[0].legend(loc='upper right')
        axes[0].tick_params(axis='x', bottom=False)
        axes[0].axhline(y=0, color='black', linewidth=1)
        axes[0].set_title(
            f'Uplift by percentile\nweighted average uplift = {uplift_weighted_avg:.4f}')

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + \
                              [f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}" for i in range(len(percentiles) - 1)]
            axes[1].set_xticks(percentiles)
            axes[1].set_xticklabels(percentiles_str, rotation=45)

        else:
            axes[1].set_xticks(percentiles)

        axes[1].legend(loc='upper right')
        axes[1].axhline(y=0, color='black', linewidth=1)
        axes[1].set_xlabel('Percentile')
        axes[1].set_title('Response rate by percentile')

    return axes