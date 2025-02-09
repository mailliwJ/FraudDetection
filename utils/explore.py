# Imports ---------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
boxprops={'facecolor':colors.to_rgba('blue', 0.4),
          'edgecolor':colors.to_rgba('black', 1),
          'linewidth':0.6,
          'linestyle':'-',
          'zorder':1}
                    
medianprops={'color':'red',
             'linewidth':1.5,
             'linestyle':'--',
             'alpha':1,
             'zorder': 2}

flierprops={'marker':'x',
            'markersize':5,
            'markerfacecolor':colors.to_rgba('blue', 1),
            'markeredgecolor':colors.to_rgba('blue', 1),
            'markeredgewidth':0.6,
            'zorder':2}
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot feature distributions by target class
def split_distributions(data=None, features=None, target='', stat='density', bins=35, kde_linewidth=0.6, normal=True, noraml_linewidth=0.6, boxplot=False):
    fig, axes = plt.subplots(6, 5, figsize=(20,15))
    axes = axes.flatten()

    for i, feature in enumerate(features):

        target_values = [val for val in data[target].unique()]
        subset_0 = data[feature].loc[data['Class'] == target_values[0]].values
        subset_1 = data[feature].loc[data['Class'] == target_values[1]].values

        sns.histplot(data=subset_0, color='green', alpha=0.4, stat=stat, element='step', bins=bins, ax=axes[i])
        sns.kdeplot(subset_0, color='green', bw_adjust=1.5, linewidth=kde_linewidth, ax=axes[i])

        sns.histplot(data=subset_1, color='#ff999b', alpha=0.5, stat=stat, element='step', bins=bins, ax=axes[i])
        sns.kdeplot(subset_1, color='red', bw_adjust=1.5, linewidth=kde_linewidth, ax=axes[i])
        
        if normal:
            mean, std = data[feature].mean(), data[feature].std()
            x = np.linspace(data[feature].min() - 10, data[feature].max() + 10, 1000)
            normal_pdf = norm.pdf(x, mean, std)
            axes[i].plot(x, normal_pdf, label='Normal Curve', color='black', linestyle='-', linewidth=noraml_linewidth)

        if boxplot:
            ax2 = axes[i].twinx()
            sns.boxplot(data=data, x=feature, ax=ax2, width=0.3, notch=True,
                        boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, showmeans=True)
        
        axes[i].set_title(f'Distribution of {feature} by Class')
        axes[i].set_ylabel('Density')
        axes[i].set_xlabel(f'{feature}')

    for j in range(len(features), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Density Distribution Plots Split by Fraud Class')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def grouped_boxplots(data=None, features=None, target='', class_names=[]):
    fig, axes = plt.subplots(6, 5, figsize=(20,15))
    axes = axes.flatten()
    palette = ['#08c37f', 'purple']
    
    for i, feature in enumerate(features):
        sns.boxplot(data, x=feature, hue=target, palette=palette, gap=0.3, ax=axes[i])
    
    for ax in axes:
        for patch in ax.patches:
            color = patch.get_facecolor()
            patch.set_facecolor((*color[:3], 0.3))

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, class_names, title=target)

    for j in range(len(features), len(axes)):
        axes[j].axis('off')

    plt.suptitle('Grouped Boxplots For Each Feature')

    plt.tight_layout()
    plt.show()

    return None

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Select features based on correlation thresholds
def select_features(data=None, target='', target_corr_threshold=0.2, collinearity_threshold=0.6):
    # Compute correlations with Class
    correlation_with_class = data.corr(numeric_only=True)[target].drop(target)

    # Filter features with correlation below a threshold (e.g., 0.2)
    high_class_corr_features = correlation_with_class[correlation_with_class.abs() >= target_corr_threshold].index.tolist()

    # Compute the correlation matrix for the filtered features
    filtered_corr_matrix = data[high_class_corr_features].corr()

    # Identify highly correlated feature pairs
    high_corr_pairs = np.where(np.abs(filtered_corr_matrix) > collinearity_threshold)
    high_corr_pairs = [(filtered_corr_matrix.index[x], filtered_corr_matrix.columns[y]) 
                    for x, y in zip(*high_corr_pairs) if x != y and x < y]

    # Step 5: Select features to keep (start with the filtered list)
    features_to_keep = high_class_corr_features.copy()
    for feature1, feature2 in high_corr_pairs:
        # Compare correlations with Class and remove the less significant feature
        if feature1 in features_to_keep and feature2 in features_to_keep:  # Check to avoid redundant removal
            if correlation_with_class[feature1] > correlation_with_class[feature2]:
                features_to_keep.remove(feature2)
            else:
                features_to_keep.remove(feature1)

    # Step 6: Create and return a DataFrame of selected features
    selected_features_df = pd.DataFrame({
        'Feature': features_to_keep,
        'CorrelationWithClass': correlation_with_class[features_to_keep]
    }).sort_values(by='CorrelationWithClass', ascending=False)

    selected_features_df.reset_index(drop=True, inplace=True)

    return (selected_features_df, features_to_keep)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

