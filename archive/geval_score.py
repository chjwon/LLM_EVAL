import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
import argparse

def calculate_correlation(pred_score, human_score, result):
    assert len(pred_score) == len(human_score)

    if len(result) == 0:
        result = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    result['pearson'] += pearsonr(pred_score, human_score)[0]
    result['spearman'] += spearmanr(pred_score, human_score)[0]
    result['kendalltau'] += kendalltau(pred_score, human_score)[0]

    return result

def print_correlations(result, n):
    if n == 0:
        n = 1
    correlations = {
        'Pearson': round(result['pearson'] / n, 4),
        'Spearman': round(result['spearman'] / n, 4),
        'Kendall Tau': round(result['kendalltau'] / n, 4)
    }
    df = pd.DataFrame(correlations, index=[0])
    print(df)

# Path to the CSV file
file_path = './geval_result.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Extract the score columns
scores = data[['coh_score', 'con_score', 'flu_score', 'rel_score']]

results = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
n = 0

# Calculate correlations for each pair of scores
for col1 in scores.columns:
    for col2 in scores.columns:
        if col1 != col2:
            results = calculate_correlation(scores[col1], scores[col2], results)
            n += 1

# Print the average correlations
print_correlations(results, n)

# Display results in a user-friendly format
results_display = pd.DataFrame([{
    'Correlation Type': 'Pearson',
    'Average Correlation': round(results['pearson'] / n, 4)
}, {
    'Correlation Type': 'Spearman',
    'Average Correlation': round(results['spearman'] / n, 4)
}, {
    'Correlation Type': 'Kendall Tau',
    'Average Correlation': round(results['kendalltau'] / n, 4)
}])

print(results_display)
