from itertools import combinations, product, chain

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation as mad
from workalendar.europe import Russia

## Basic functions


def squeeze_df(df, numeric_only=True, ignore_cols=None):
    """Reduce memory usage and encode categorical features."""

    if ignore_cols is None:
        cols = dict(df.dtypes)
    else:
        cols = dict(df.drop(ignore_cols, axis=1, errors='ignore').dtypes)

    if numeric_only:
        for col, t in cols.items():
            if 'float' in str(t):
                df[col] = pd.to_numeric(df[col], downcast='float')
            if 'int' in str(t):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif t == 'object':
                pass
    else:
        for col, t in cols.items():
            if 'float' in str(t):
                df[col] = pd.to_numeric(df[col], downcast='float')
            if 'int' in str(t):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif t == 'object':
                df[col] = df[col].astype(pd.CategoricalDtype(ordered=True))

    return df


## Calendar variables


def get_week_of_month(dates):
    first_day = dates - pd.to_timedelta(dates.dt.day - 1, unit='d')
    return (dates.dt.day - 1 + first_day.dt.weekday) // 7 + 1


def calendar_vars(df, date_col='week', default_suffix='date', year_col=False, day_col=False,
                  get_business_days=True, daily_data=False):
    """Creating calendar variables.

    Parameters
    ----------
    date_col : the primary date column (either week or day column).

    daily_data : aggregates the final result by day / week.
    """

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    date_range = pd.date_range(start=min_date - pd.Timedelta(2, 'W'), end=max_date + pd.Timedelta(2, 'W'), freq='d')
    days_weeks = pd.DataFrame({'day': date_range, 'week': [x.to_period('W').start_time for x in date_range]})

    # Basic calendar variables

    if year_col:
        days_weeks[f'{default_suffix}_year'] = days_weeks['day'].dt.year  # .isocalendar().year
    if day_col:
        days_weeks[f'{default_suffix}_day'] = days_weeks['day'].dt.day
    if daily_data:
        days_weeks[f'{default_suffix}_week_year'] = [x.isocalendar()[1] for x in days_weeks['day']]
    else:
        days_weeks[f'{default_suffix}_week_year'] = [x.isocalendar()[1] for x in days_weeks['week']]

    days_weeks[f'{default_suffix}_month'] = days_weeks['day'].dt.month
    days_weeks[f'{default_suffix}_w_of_m_day'] = get_week_of_month(days_weeks['day'])
    days_weeks[f'{default_suffix}_w_of_m_week'] = ((days_weeks['week'].dt.day - 1) // 7 + 1)
    days_weeks[f'{default_suffix}_weekday'] = days_weeks['day'].dt.weekday + 1

    # Week of month shares (the beginning or the end of a month)

    days_weeks['working_week'] = [1 if x < 6 else 0 for x in days_weeks[f'{default_suffix}_weekday']]
    days_weeks_no_weekends = days_weeks.loc[days_weeks['working_week'] == 1].copy()

    days_weeks_no_weekends['num'] = days_weeks_no_weekends.groupby(['week', f'{default_suffix}_w_of_m_day'],
                                                                   group_keys=False)[f'{default_suffix}_w_of_m_day'].transform(pd.Series.count)
    days_weeks_no_weekends['denom'] = days_weeks_no_weekends.groupby(['week'], as_index=False)[f'{default_suffix}_w_of_m_week'].transform(pd.Series.count)
    days_weeks_no_weekends['share'] = days_weeks_no_weekends['num'] / days_weeks_no_weekends['denom']

    days_weeks_no_weekends = days_weeks_no_weekends.groupby('week', as_index=False)['share'].max()
    days_weeks = days_weeks.merge(days_weeks_no_weekends, how='left', on='week')
    days_weeks = days_weeks.drop('working_week', axis=1)

    if get_business_days:
        cal = Russia()

        days_weeks['business_day'] = [cal.is_working_day(x) for x in days_weeks['day']]
        days_weeks['business_day'] = [1 if x == True else 0 for x in days_weeks['business_day']]
        days_weeks['business_week'] = days_weeks.groupby(['week'], group_keys=False)['business_day'].transform(np.sum)
        col_business_dates = ['business_day']
    else:
        col_business_dates = []

    # Aggregating data by day / week

    if daily_data:
        days_weeks = days_weeks.drop(['week', 'date_w_of_m_week', 'share'], axis=1)
        df = df.merge(days_weeks, how='left', on='day')
    else:
        days_weeks = days_weeks.drop(['day', 'date_w_of_m_day', 'date_weekday'] + col_business_dates, axis=1).copy()
        days_weeks = days_weeks.drop_duplicates(subset=['week'])
        df = df.merge(days_weeks, how='left', on='week')

    return df


def holiday_vars(df, fc_horizon_weeks=10, date_col='week'):
    """Creating holiday variables, including lag holiday variables."""

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    date_range = pd.date_range(start=min_date - pd.Timedelta(2, 'W'), end=max_date + pd.Timedelta(fc_horizon_weeks + 2, 'W'), freq='d')
    days_weeks = pd.DataFrame({'day': date_range, 'week': [x.to_period('W').start_time for x in date_range]})
    days_weeks['year'] = days_weeks['day'].dt.year

    # Basic holiday variables

    all_holidays = []

    cal = Russia()

    for x in days_weeks['year'].unique():
        all_holidays.append(cal.holidays(x))

    all_holidays = list(chain(*all_holidays))
    all_holidays = pd.DataFrame(all_holidays, columns=['day', 'holiday'])
    all_holidays['day'] = pd.to_datetime(all_holidays['day'])

    all_holidays['remove'] = [1 if ('after' in x) or ('After' in x) or ('shift' in x) or
                             ('Before' in x) or ('before' in x) else 0 for x in all_holidays['holiday']]
    all_holidays = all_holidays.loc[all_holidays['remove'] == 0].drop('remove', axis=1).copy()

    all_holidays['week'] = [x.to_period('W').start_time for x in all_holidays['day']]

    all_holidays['holiday'] = [np.nan if x == 'no' else x for x in all_holidays['holiday']]

    if date_col == 'week':
        days_weeks = days_weeks.drop_duplicates(subset=['week']).drop(['day', 'year'], axis=1).copy()
        days_weeks = days_weeks.merge(all_holidays[['week', 'holiday']], how='left', on='week')

        # New Year's inconsistencies

        days_weeks = days_weeks.loc[days_weeks['holiday'] != "New Year's Eve"].copy()
        days_weeks['month'] = days_weeks['week'].dt.month
        days_weeks['week'] = [x - pd.Timedelta(1, 'W') if y == 1 and z == 'New year' else x for x, y, z in
                              zip(days_weeks['week'], days_weeks['month'], days_weeks['holiday'])]

        days_weeks['duplicates'] = days_weeks.duplicated(subset='week', keep=False)
        days_weeks['duplicates'] = [0 if dup is True and pd.isnull(holiday) else 1 for dup, holiday in
                                    zip(days_weeks['duplicates'], days_weeks['holiday'])]
        days_weeks = days_weeks.loc[days_weeks['duplicates'] == 1].drop(['month', 'duplicates'], axis=1).copy()

    else:
        days_weeks = days_weeks.drop(['week', 'year'], axis=1).copy()
        days_weeks = days_weeks.merge(all_holidays[['day', 'holiday']], how='left', on='day')

    # Lag variables

    if date_col == 'week':
        lags = range(1, 4)
    else:
        lags = range(1, 3*7 + 1)

    for x in lags:
        days_weeks[f'holiday_lag_{x}+'] = days_weeks['holiday'].shift(x)
        days_weeks[f'holiday_lag_{x}-'] = days_weeks['holiday'].shift(-x)

    days_weeks = days_weeks.dropna(subset=days_weeks.columns.difference([date_col]),
                                    how='all', axis=0).reset_index(drop=True)

    return days_weeks


## Encoding techniques


def target_enc_vars(df, week_col='week', testing=False, n_choose_1={}, n_choose_2={}, n_choose_2_drop={},
                    methods=['mean'], target_var='QuantityDal', fc_start=None, last_years='all'):
    """Creating target-encoded variables.

    Parameters
    ----------
    week_col : needed only to trim dataset for the testing (CV) phase.

    n_choose_1 : single-level categorical variables. For each group the target mean / std is calculated.
    For instance, the target encoded variable for 'month' is:
    df.groupby(['month'], group_keys=False, observed=True)['Sales'].transform(np.mean)

    n_choose_2 : combinations of categorical variables. For each group the target mean / std is calculated.
    For instance, the target encoded variable for 'month' and 'cluster' is:
    df.groupby(['month', 'cluster'], group_keys=False, observed=True)['Sales'].transform(np.mean)

    n_choose_2_drop : combinations of categorical variables that must be removed.

    last_years : the number of years used for mean / std calculation.
    """

    df_c = df.copy()

    if testing:
        df_c[target_var] = [np.nan if x >= fc_start else y for
                            x, y in zip(df_c[week_col], df_c[target_var])]

    if last_years == 'all':
        pass
    else:
        df_c['year'] = df_c[week_col].dt.year
        list_years = df_c['year'].unique()
        list_years.sort()
        list_years = list_years[-last_years:]
        df_c[target_var] = [y if x in list_years else np.nan for x, y in zip(df_c['year'], df_c[target_var])]

    total_combs = []

    for key, value in n_choose_1.items():
        total_combs.append(list(combinations(value, 1)))

    for key, value in n_choose_2.items():
        total_combs.append(list(combinations(value, 2)))

    total_combs = [item for sublist in total_combs for item in sublist]

    drop_combs = list(combinations(n_choose_2_drop, 2))
    total_combs = list(set(total_combs) - set(drop_combs))

    enc_train, enc_test, col_names = [], [], []

    for method in methods:
        for comb in total_combs:
            col_name = f'enc_{"_".join(comb)}_{method}_{last_years}'
            col_names.append(col_name)

            # Training set

            encoded_tr = pd.to_numeric(df_c.groupby(list(comb), group_keys=False,
                                                    observed=True)[target_var].transform(method), downcast='float')

            enc_train.append(encoded_tr)

            # Test set

            encoded_te = df_c.groupby(list(comb), group_keys=False, observed=True,
                                      as_index=False).aggregate({target_var: method}).rename(columns={target_var: col_name})

            enc_test.append(encoded_te)

    enc_train = pd.DataFrame(enc_train, index=col_names).T

    return total_combs, col_names, enc_train, enc_test


def target_enc_vars_test(df, df_target_train):
    """Creating target-encoded variables for the test set."""

    total_combs = int((len(df_target_train[1]) / len(df_target_train[0]))) * df_target_train[0]

    for idx, comb in enumerate(total_combs):
        df = pd.merge(df, df_target_train[3][idx], on=comb, how='left')

    return df


## Lag variables


def lag_vars(df, week_col='week', testing=False,
             n_lags=13, specific_lags=[51, 52, 53], rolls=[2, 4, 6, 8], rolls_lags=[1, 2, 3, 4],  exp_mean=True,
             target_var='QuantityDal', id_var='ID', fc_start=None):
    """Creating lag variables.

    Parameters
    ----------
    specific_lags : adding hand-picked lag variables.

    rolls : size of the moving window.

    rolls_lags : specific lags for moving averages.
    """

    df_c = df.copy()

    if testing:
        df_c[target_var] = [np.nan if x >= fc_start else y for
                            x, y in zip(df_c[week_col], df_c[target_var])]

    df_lags, col_names = [], []

    vol_group_0 = df_c.groupby([id_var], observed=True, group_keys=False)[target_var]

    for x in range(1, n_lags + 1):
        col_name = f'lag_{x}'
        col_names.append(col_name)
        lag_var = pd.to_numeric(vol_group_0.shift(x), downcast='float')
        df_lags.append(lag_var)

    for x in specific_lags:
        col_name = f'lag_{x}'
        col_names.append(col_name)
        lag_var = pd.to_numeric(vol_group_0.shift(x), downcast='float')
        df_lags.append(lag_var)

    for x in product(rolls, rolls_lags):
        col_name_0 = f'lag_{x[1]}_roll_{x[0]}_mean'
        col_names.append(col_name_0)
        lag_var_0 = pd.to_numeric(vol_group_0.shift(x[1]).rolling(x[0]).mean(), downcast='float')
        df_lags.append(lag_var_0)

        col_name_1 = f'lag_{x[1]}_roll_{x[0]}_std'
        col_names.append(col_name_1)
        lag_var_1 = pd.to_numeric(vol_group_0.shift(x[1]).rolling(x[0]).std(), downcast='float')
        df_lags.append(lag_var_1)

    for y in rolls_lags:
        col_name = f'lag_{y}_expand_mean'
        col_names.append(col_name)
        lag_var = pd.to_numeric(vol_group_0.apply(lambda x: x.expanding(4).mean().shift(y)), downcast='float')
        df_lags.append(lag_var)

    if exp_mean:
        for y in rolls_lags:
            col_name_0 = f'lag_{y}_exponent_mean_0.25'
            col_names.append(col_name_0)
            lag_var_0 = pd.to_numeric(vol_group_0.apply(lambda x: x.ewm(alpha=0.25).mean().shift(y)), downcast='float')
            df_lags.append(lag_var_0)

            col_name_1 = f'lag_{y}_exponent_mean_0.70'
            col_names.append(col_name_1)
            lag_var_1 = pd.to_numeric(vol_group_0.apply(lambda x: x.ewm(alpha=0.7).mean().shift(y)), downcast='float')
            df_lags.append(lag_var_1)

    df_lags = pd.DataFrame(df_lags, index=col_names).T

    return df_lags


if __name__ == '__main__':
    squeeze_df()
    calendar_vars()
    holiday_vars()
    target_enc_vars()
    target_enc_vars_test()
    lag_vars()