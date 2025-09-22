"""
This script simulates a baseline model for the infodengue-mosqlimate challenge; the model is a negative binomial distribution fit to the weekly dengue incidence from 2010-2025
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, Bento Lab, Cornell University CVM Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
from epiweeks import Week
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import nbinom
from scipy.optimize import minimize


##############
## settings ##
##############

validation_idx = None  # None (forecast), 1, 2 or 3 (validation)
end_train_epiweek = 25
start_predict_epiweek = 41
end_predict_epiweek = 40

if validation_idx:
    # check user input
    assert validation_idx in [1,2,3], "'validation' must be equal to 1, 2 or 3."
    # set ID and training dates for validation
    ID = f'validation_{validation_idx}'
    end_train_year = 2021 + validation_idx
    start_predict_year = 2021 + validation_idx
    end_predict_year = 2021 + validation_idx + 1
else:
    # set ID and training dates for forecast 
    ID = f'forecast'
    validation_idx = 4
    end_train_year = 2021 + validation_idx
    start_predict_year = 2021 + validation_idx
    end_predict_year = 2021 + validation_idx + 1

# desired quantiles
quantiles = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]


#############################################
## model parameter estimation & prediction ##
#############################################

# log likelihood
def negbinom_log_likelihood(params, x, weights):
    # unpack parameters
    r, p = params
    # keep them in valid range
    if r <= 0 or p <= 0 or p >= 1:
        return np.inf
    return -np.sum(weights * nbinom.logpmf(x, r, p))

# load data
data = pd.read_csv('../data/raw/dengue.csv', dtype={'epiweek': str})

# aggregate by UF
data = data[['epiweek', 'uf', 'casos']].groupby(by=['epiweek', 'uf']).sum().reset_index()

# split epiweek year and month
data['epiweek_year'] = data['epiweek'].apply(lambda x: int(x[:-2]))
data['epiweek_week'] = data['epiweek'].apply(lambda x: int(x[-2:]))
data = data[['epiweek_year', 'epiweek_week', 'uf', 'casos']]

# filter out until end_train_year + end_train_epiweek
data = data[(~(data['epiweek_year'] > end_train_year) & (  ~((data['epiweek_year'] == end_train_year) & (data['epiweek_week'] > end_train_epiweek)) ))]

# compute unique epiweek/ufs
epiweek_weeks = data['epiweek_week'].unique().tolist()
ufs = data['uf'].unique().tolist()

# fit model and gather results
results= []
for uf in ufs:
    for epiweek_week in epiweek_weeks:
        # get data
        subset = data[((data['uf'] == uf) & (data['epiweek_week'] == epiweek_week))]
        counts = subset['casos']
        years = subset['epiweek_year'].values
        # parametrise neg. expon. weighing of historical seasons
        lambda_ = 0.0 # uniform weighing
        max_year = subset['epiweek_year'].max()
        weights = np.exp(-lambda_ * (max_year - years))
        weights = weights / weights.sum()
        # method of moments estimate for r and p as initial guess
        mean_x = np.mean(counts)
        var_x = np.var(counts, ddof=1)
        p0 = mean_x / var_x
        r0 = mean_x * p0 / (1 - p0)
        p0 = max(min(p0, 0.999), 0.001)  # keep p0 in valid range
        r0 = max(r0, 10)
        # fit distribution
        res = minimize(lambda params: negbinom_log_likelihood(params, counts, weights),
                       x0=[r0, p0], bounds=[(1e-9, None), (1e-9, 1-1e-9)])
        r_hat, p_hat = res.x
        # simulate desired quantiles
        q_values = nbinom.ppf(quantiles, r_hat, p_hat)
        quantile_df = pd.DataFrame({'quantile': quantiles, 'casos': q_values})
        # attach the uf + epiweek
        quantile_df['uf'] = uf
        quantile_df['epiweek_week'] = epiweek_week
        # fill any nan with zero
        quantile_df = quantile_df.fillna(0)
        # append result
        results.append(quantile_df)
# concatenate all fitted distributions    
results = pd.concat(results, axis=0)


#################################################
## convert to desired format for the challenge ##
#################################################

# convert start- and end epiweek of the "prediction" to a daterange
start_date = pd.to_datetime(f'{start_predict_year}-W{start_predict_epiweek-1:02d}-1', format='%G-W%V-%u')
end_date = pd.to_datetime(f'{end_predict_year}-W{end_predict_epiweek:02d}-1', format='%G-W%V-%u')
dates = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
# pre-allocate an output dataframe containing the cartesian product of all spatial units and dates as index & the quantiles/median as columns
index = pd.MultiIndex.from_product([dates, ufs], names=['date', 'adm_1'])
output = pd.DataFrame(index=index, columns=['lower_95', 'upper_95', 'lower_90', 'upper_90', 'lower_80', 'upper_80', 'lower_50', 'upper_50', 'pred'])
output = output.reset_index()
output['epiweek_week'] = output['date'].apply(lambda x: int(str(Week.fromdate(x).year * 100 + Week.fromdate(x).week)[-2:]))
# fill the dataframe
for ew in output['epiweek_week'].unique():
    for uf in output['adm_1'].unique():
        # Median
        output.loc[((output['epiweek_week'] == ew) & (output['adm_1'] == uf)), 'pred'] = results.loc[((results['uf'] == uf) & (results['epiweek_week'] == ew) & (results['quantile'] == 0.5)), 'casos'].values
        # 95% confint
        output.loc[((output['epiweek_week'] == ew) & (output['adm_1'] == uf)), 'lower_95'] = results.loc[((results['uf'] == uf) & (results['epiweek_week'] == ew) & (results['quantile'] == 0.025)), 'casos'].values
        output.loc[((output['epiweek_week'] == ew) & (output['adm_1'] == uf)), 'upper_95'] = results.loc[((results['uf'] == uf) & (results['epiweek_week'] == ew) & (results['quantile'] == 0.975)), 'casos'].values
        # 90% confint
        output.loc[((output['epiweek_week'] == ew) & (output['adm_1'] == uf)), 'lower_90'] = results.loc[((results['uf'] == uf) & (results['epiweek_week'] == ew) & (results['quantile'] == 0.05)), 'casos'].values
        output.loc[((output['epiweek_week'] == ew) & (output['adm_1'] == uf)), 'upper_90'] = results.loc[((results['uf'] == uf) & (results['epiweek_week'] == ew) & (results['quantile'] == 0.95)), 'casos'].values
        # 80% confint
        output.loc[((output['epiweek_week'] == ew) & (output['adm_1'] == uf)), 'lower_80'] = results.loc[((results['uf'] == uf) & (results['epiweek_week'] == ew) & (results['quantile'] == 0.10)), 'casos'].values
        output.loc[((output['epiweek_week'] == ew) & (output['adm_1'] == uf)), 'upper_80'] = results.loc[((results['uf'] == uf) & (results['epiweek_week'] == ew) & (results['quantile'] == 0.90)), 'casos'].values
        # 50% confint
        output.loc[((output['epiweek_week'] == ew) & (output['adm_1'] == uf)), 'lower_50'] = results.loc[((results['uf'] == uf) & (results['epiweek_week'] == ew) & (results['quantile'] == 0.25)), 'casos'].values
        output.loc[((output['epiweek_week'] == ew) & (output['adm_1'] == uf)), 'upper_50'] = results.loc[((results['uf'] == uf) & (results['epiweek_week'] == ew) & (results['quantile'] == 0.75)), 'casos'].values
# remove 'epiweek_week' column
output = output.drop('epiweek_week', axis=1)
# save the result
output.to_csv(f'../data/interim/baseline_model-{ID}.csv')


##########################
## save a visualisation ##
##########################

# Make the folder structure to save results
## define figure path
fig_path=f'../data/interim/fig/{ID}/' # Path to backend
## check if samples folder exists, if not, make it
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# plot result per uf
for uf in ufs:
    median = []
    ll_50 = []
    ul_50 = []
    ll_95 = []
    ul_95 = []
    fig,ax=plt.subplots(figsize=(11.7, 8.3/3))
    for epiweek_week in epiweek_weeks:
        # get data
        x = data[((data['uf'] == uf) & (data['epiweek_week'] == epiweek_week))]['casos']
        # plot data
        ax.scatter(int(epiweek_week) * np.ones(len(x)), x, color='black', alpha=0.2, facecolors='none')
        # get 50% and 95% quantiles
        median.append(int(results[((results['uf'] == uf) & (results['epiweek_week'] == epiweek_week) & (results['quantile'] == 0.50))]['casos'].values))
        ll_50.append(int(results[((results['uf'] == uf) & (results['epiweek_week'] == epiweek_week) & (results['quantile'] == 0.25))]['casos'].values))
        ul_50.append(int(results[((results['uf'] == uf) & (results['epiweek_week'] == epiweek_week) & (results['quantile'] == 0.75))]['casos'].values))
        ll_95.append(int(results[((results['uf'] == uf) & (results['epiweek_week'] == epiweek_week) & (results['quantile'] == 0.025))]['casos'].values))
        ul_95.append(int(results[((results['uf'] == uf) & (results['epiweek_week'] == epiweek_week) & (results['quantile'] == 0.975))]['casos'].values))
    # visualise quantiles
    ax.fill_between(epiweek_weeks, ll_50, ul_50, color='red', alpha=0.3)
    ax.fill_between(epiweek_weeks, ll_95, ul_95, color='red', alpha=0.1)
    # visualise median
    ax.plot(epiweek_weeks, median, color='red', linestyle='--')
    # make figure pretty
    ax.set_title(f'{uf}')
    ax.set_xlabel('CDC epiweek (-)')
    ax.set_ylabel('DENV inc. (per week)')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    plt.tight_layout()
    plt.savefig(fig_path+f'{uf}.pdf')
    plt.close()

