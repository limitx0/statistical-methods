

from ucimlrepo import fetch_ucirepo
import rdata

from pprint import pprint
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from scipy.stats import spearmanr


def brief_description(print_description=False):
    if print_description:
        print('''
            The National Poll on Healthy Aging (NPHA) is a recurring household survey that measures the perspectives and experiences of adults aged 50-80 in the United States on a wide range of health and health policy issues. 
            The survey is conducted by the University of Michigan Institute for Healthcare Policy and Innovation, sponsored by AARP and Michigan Medicine, and designed and analyzed by the University of Michigan's A. Alfred Taubman Medical Research Institute. 
            The NPHA is designed to inform the public, health care providers, policymakers, and advocates on issues related to health, health care, and health policy for older adults. 
            ''')

    # https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)
    # fetch dataset
    npha = fetch_ucirepo(id=936)
    
    # print(f'\n\nnpha=\n{npha}')
    # print(f'\n\nnpha.data=\n{npha.data}')

    if print_description:
        # data (as pandas dataframes)
        X = npha.data.features
        y = npha.data.targets

        # metadata
        pprint(npha.metadata)

        # variable information
        print(npha.variables)
    
    
    print('features\' unique values')
    for col in npha.data.features.columns:
        unique_values = npha.data.features[col].unique()
        print(f'{col}={unique_values}')
    print('\n========================================\n\n')
    
    return npha


def original_questionnaire_data(print_description=False):
    original_data = rdata.read_rda(
        fr'.\ICPSR_37305-V1\ICPSR_37305\DS0001\37305-0001-Data.rda')
    
    if print_description:
        print(f'original_data=\n{original_data}')
        
        print()
        for col in original_data.columns:
            print(f'{col}={original_data[col].unique()}')
    
    print('\n========================================\n\n')
    
    return original_data
    
    

# Statistical Methods
'''
Parametric tests
Non-parametric tests
Types of Data: Categorical Data (ordinal scale)
Types of Research Questions: 
    Relationships: Questions that involve examining the relationship between variables.
    Frequency Counts: Questions that involve comparing counts or proportions.


Roadmap for Choosing a Statistical Test
1. Determine the Type of Data
    - Numerical (Interval or Ratio): Data with meaningful order and a known unit of measurement.
    - Categorical (Nominal or Ordinal): Data representing groups or categories.
2. Assess the Study Design
    - Number of Samples: Single, two, or multiple groups.
    - Related or Independent Samples: Whether the samples are paired/related (e.g., measurements before and after an intervention on the same subjects) or independent from each other.
3. Check for Distribution Assumptions
    - Normality: Whether the data are normally distributed (parametric tests) or not (non-parametric alternatives).
4. Decide on the Hypothesis Type
    - One-Tailed vs. Two-Tailed Tests: Whether the hypothesis is directional (one-tailed) or non-directional (two-tailed).


Common Statistical Tests
 • For Categorical Data:
    - Chi-Squared Test of Independence: Tests if there is a relationship between two categorical variables.
    - Fisher's Exact Test: Used instead of the Chi-Squared test when sample sizes are small.
 • For Examining Relationships:
    XX - Pearson's Correlation: Measures the strength of a linear relationship between two continuous variables.
    - Spearman's Rank Correlation: Non-parametric measure of the strength of a monotonic relationship between two variables.


Key Concepts in Nonparametric Statistics
 • Distribution-Free Methods
 • Rank-Based Tests
 • Permutation Tests
 • Sign Tests

'''


def chi2_test_of_independence(feat_row, feat_col, alpha=0.05):
    '''
    Chi-Squared Test of Independence
    '''
    
    print('Chi-Squared Test of Independence')
    
    # create the contingency table
    feat_row_uni = sorted(feat_row.unique())
    feat_col_uni = sorted(feat_col.unique())
    
    observed = []
    for row in feat_row_uni:
        row_obs = []
        for col in feat_col_uni:
            row_obs.append(((feat_row == row) & (feat_col == col)).sum())
        observed.append(row_obs)
        
    # print the contingency table
    print('\nobserved=')
    print(f'row: {feat_row.name}')
    print(f'col: {feat_col.name}')
    
    observed_row_names = [f'{val}' for val in feat_row_uni]
    observed_col_names = [f'{val}' for val in feat_col_uni]
    for col_name in observed_col_names:
        print(f'\t{col_name}', end='')
    print()
    for row_name, row in zip(observed_row_names, observed):
        print(f'{row_name}: ', end='')
        for row_elm in row:
            print(f'\t{row_elm}', end='')
        print()
    print()
    
    
    # print results
    chi2, p, dof, expected = chi2_contingency(observed)
    print("chi2:", chi2) # The test statistic.
    print("p-value:", p) # The p-value of the test.
    print("dof:", dof) # The degrees of freedom.
    print("expected=\n", expected) # The expected frequencies, based on the marginal sums of the table.
    
    # statistics inference
    print('\nInference:')
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    
    
def spearman_rank_corr(x, y):
    '''
    Spearman's rank correlation is a non-parametric measure of the strength and direction of association between two ranked variables.
    
    Spearman's correlation coefficient:
        +1 indicates a perfect positive monotonic relationship
        -1 indicates a perfect negative monotonic relationship
        0 suggests no monotonic relationship.

    p-value:
        The p-value for a hypothesis test whose null hypothesis is that two samples have no ordinal correlation. 
        See alternative above for alternative hypotheses. 
        pvalue has the same shape as statistic.
    '''
    # https://www.geeksforgeeks.org/spearmans-rank-correlation/
    # https://stats.stackexchange.com/questions/55288/understanding-the-p-value-in-spearmans-rank-correlation
    
    print('Spearman\'s Rank Correlation')
    
    print(f'x: {x.name}')
    print(f'y: {y.name}')
    
    corr, pval = spearmanr(x, y)
    
    # draw scatter plot
    plt.scatter(x, y, color='blue', marker='o', s=30, alpha=0.05)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.title(f'{x.name} vs {y.name}')
    plt.show()
    # plt.savefig()
    
    # print results
    print("Spearman's correlation coefficient:", corr)
    print("p-value:", pval)
    

    


if __name__ == "__main__":
    # original = original_questionnaire_data()['da37305.0001']
    npha = brief_description()


    Physical_Health = npha.data.features['Physical_Health']
    Mental_Health = npha.data.features['Mental_Health']
    spearman_rank_corr(Physical_Health, Mental_Health)

    
    Number_of_Doctors_Visited = npha.data.targets['Number_of_Doctors_Visited']
    chi2_test_of_independence(Physical_Health, Number_of_Doctors_Visited)
    
    
    

