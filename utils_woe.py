#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def bin_and_plot_woe_manual(data, feature, target, bins=5, eps=0.0001):
    """
    Bin a continuous variable using equal-width binning (pd.cut) and plot WOE.
    
    Args:
        data (pd.DataFrame): The dataset containing the feature and target.
        feature (str): The name of the continuous feature.
        target (str): The binary target variable.
        bins (int): Number of bins for pd.cut().
        eps (float): Small value to prevent division by zero.
    
    Returns:
        woe_df (pd.DataFrame): DataFrame with bins and WOE values.
    """
    # Create bins using pd.cut
    data['bin'] = pd.cut(data[feature], bins=bins, include_lowest=True, duplicates='drop')

    # Aggregate event & non-event counts
    woe_df = data.groupby('bin', observed=True).agg(
        total_count=(target, 'count'),
        event_count=(target, 'sum')
    ).reset_index()

    woe_df['non_event_count'] = woe_df['total_count'] - woe_df['event_count']

    # Compute event and non-event rates (avoid division by zero)
    woe_df['event_rate'] = (woe_df['event_count'] + eps) / woe_df['event_count'].sum()
    woe_df['non_event_rate'] = (woe_df['non_event_count'] + eps) / woe_df['non_event_count'].sum()
    
    # Compute WOE
    woe_df['WOE'] = round(np.log(woe_df['non_event_rate'] / woe_df['event_rate']),4)

     # Compute IV for each bin
    woe_df['IV'] = round((woe_df['non_event_rate'] - woe_df['event_rate']) * woe_df['WOE'],4)
    
    # Compute total IV
    total_IV = woe_df['IV'].sum()
    print(f'Total IV for {feature}: {total_IV:.4f}')
    
    # Convert bins to string format for plotting
    # woe_df['bin_str'] = woe_df['bin'].astype(str)

    # Plot WOE distribution
    #plot_woe_distribution(woe_df, feature)
    
    return woe_df


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from decimal import Decimal, ROUND_DOWN

def truncate(value, decimals=3):
    factor = Decimal('1.' + '0' * decimals)
    return float(Decimal(value).quantize(factor, rounding=ROUND_DOWN))


def bin_and_plot_woe_tree(data, feature, target, max_depth=4, min_samples_leaf=2000, eps=0.0001):
    """
    Bin a continuous variable using DecisionTreeClassifier and compute WOE.
    
    Args:
        data (pd.DataFrame): The dataset containing the feature and target.
        feature (str): The name of the continuous feature.
        target (str): The binary target variable.
        max_depth (int): Max depth of the decision tree for binning.
        min_samples_leaf (int): Minimum samples per leaf to avoid overfitting.
        eps (float): Small value to prevent division by zero.
    
    Returns:
        woe_df (pd.DataFrame): DataFrame with bins, WOE values, and IV.
    """
    # Fit Decision Tree for binning
    X = data[[feature]]
    y = data[target]
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    tree.fit(X, y)
    
    # Get bin edges from decision tree and truncate to 3 decimal places
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]  # Remove leaf node markers
    thresholds = sorted(thresholds)
    thresholds = [truncate(th, 3) for th in thresholds]
    
    # Define bin edges
    bins = [truncate(data[feature].min(), 3)] + thresholds + [truncate(data[feature].max(), 3)]
    bins[0] = truncate(data[feature].min(), 3)
    
    # Bin the data
    data['bin'] = pd.cut(data[feature], bins=bins, include_lowest=True)

    # Aggregate event & non-event counts
    woe_df = data.groupby('bin').agg(
        total_count=(target, 'count'),
        event_count=(target, 'sum')
    ).reset_index()
    woe_df['non_event_count'] = woe_df['total_count'] - woe_df['event_count']
    
    # Compute event and non-event rates (avoid division by zero)
    woe_df['event_rate'] = (woe_df['event_count'] + eps) / woe_df['event_count'].sum()
    woe_df['non_event_rate'] = (woe_df['non_event_count'] + eps) / woe_df['non_event_count'].sum()
    
    # Compute WOE
    woe_df['WOE'] = np.log(woe_df['non_event_rate'] / woe_df['event_rate'])
    
    # Compute IV for each bin
    woe_df['IV'] = (woe_df['non_event_rate'] - woe_df['event_rate']) * woe_df['WOE']
    
    # Compute total IV
    total_IV = woe_df['IV'].sum()
    print(f'Total IV for {feature}: {total_IV:.4f}')
    
    # Plot WOE distribution
    #plot_woe_distribution(woe_df, feature)
    
    return woe_df