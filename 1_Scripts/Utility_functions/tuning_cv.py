import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import mean_tweedie_deviance, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score

## CV, loss functions


def custom_cv(df, week_col='week', val_end_week='2022-01-25', val_size_week=8, n_splits_week=4):
    """Custom cv for time-series data."""

    val_end_week = pd.to_datetime(val_end_week)
    cv_idx = []

    for x in range(n_splits_week, 0, -1): # range(start, stop, step)
        tr_threshold = val_end_week - pd.to_timedelta(val_size_week * x, unit='W')
        val_threshold = tr_threshold + pd.to_timedelta(val_size_week, unit='W')

        tr_idx = np.array(df.index[df[week_col] <= tr_threshold])
        te_idx = np.array(df.index[(df[week_col] > tr_threshold) & (df[week_col] <= val_threshold)])

        cv_idx.append((tr_idx, te_idx))

    return cv_idx


def custom_mean_tweedie_deviance(y_true, y_pred, sample_weight=None, power=0):
    """Mean Tweedie deviance regression loss."""

    # Custom part

    y_pred[y_pred < 0] = 0
    y_true = np.squeeze(np.array(y_true))

    if np.squeeze(np.array(y_true)).shape != y_pred.shape:
        sys.exit('Incorrect shapes of arrays')

    # Sklearn default

    p = power
    if p < 0:
        # 'Extreme stable', y any real number, y_pred > 0
        dev = 2 * (
            np.power(np.maximum(y_true, 0), 2 - p) / ((1 - p) * (2 - p))
            - y_true * np.power(y_pred, 1 - p) / (1 - p)
            + np.power(y_pred, 2 - p) / (2 - p))

    elif p == 0:
        # Normal distribution, y and y_pred any real number
        dev = (y_true - y_pred) ** 2

    elif p == 1:
        # Poisson distribution
        dev = 2 * (xlogy(y_true, y_true / y_pred) - y_true + y_pred)

    elif p == 2:
        # Gamma distribution
        dev = 2 * (np.log(y_pred / y_true) + y_true / y_pred - 1)

    else:
        dev = 2 * (
            np.power(y_true, 2 - p) / ((1 - p) * (2 - p))
            - y_true * np.power(y_pred, 1 - p) / (1 - p)
            + np.power(y_pred, 2 - p) / (2 - p))

    return np.average(dev, weights = sample_weight)


## Optuna


def func_to_minimise_lgb(model, x_train, y_train, x_val_es, y_val_es,
                         cv, optimised_value='cv_rmse', suppress_warnings=True):
    """Possible types of optimised_value: 'hold_out_rmse', 'cv_rmse', 'hold_out_tweedie', 'cv_tweedie'.
    CV is more time-consuming but should be more representative of the general population."""

    if suppress_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    lgb_model = model.fit(x_train, y_train,
                          callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(100)],  # verbose=0,
                          eval_set=[[x_val_es, y_val_es]])

    best_iter = lgb_model.booster_.best_iteration

    if optimised_value == 'cv_rmse':
        params_upd = lgb_model.get_params()
        params_upd['n_estimators'] = lgb_model.booster_.best_iteration

        model_upd = lgb.LGBMRegressor(**params_upd)

        lgb_CV_scores = cross_val_score(estimator=model_upd, X=x_train, y=y_train,
                                        scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)

        objective_function = -1*np.mean(lgb_CV_scores)
    
    elif optimised_value == 'cv_tweedie':
        params_upd = lgb_model.get_params()
        params_upd['n_estimators'] = lgb_model.booster_.best_iteration

        model_upd = lgb.LGBMRegressor(**params_upd)

        power = lgb_model.get_params()['tweedie_variance_power']
        mtd_scorer = make_scorer(custom_mean_tweedie_deviance, greater_is_better=True, power=power)

        lgb_CV_scores = cross_val_score(estimator=model_upd, X=x_train, y=y_train,
                                        scoring=mtd_scorer, cv=cv, n_jobs=-1)

        objective_function = np.mean(lgb_CV_scores)
    
    return objective_function, best_iter


class objective_lgb:
    def __init__(self, optimised_value, x_train, y_train, x_val_es, y_val_es,
                 cv, suppress_warnings, daily_data):
        """Bla.bla."""

        self.optimised_value = optimised_value
        self.x_train = x_train
        self.y_train = y_train
        self.x_val_es = x_val_es
        self.y_val_es = y_val_es
        self.cv = cv
        self.suppress_warnings = suppress_warnings

        self.daily_data = daily_data

    def __call__(self, trial):
        if self.daily_data:
            params = {'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
                      'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),  # min_data_in_leaf
                      'num_leaves': trial.suggest_int('num_leaves', 40, 800),
                      'max_depth': trial.suggest_int('max_depth', 5, 20),
                      # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 4),  # lambda_l1
                      # 'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 4),  # lambda_l2
                      'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)}  # feature_fraction

        else:
            params = {'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
                      'min_child_samples': trial.suggest_int('min_child_samples', 3, 80),  # min_data_in_leaf
                      'num_leaves': trial.suggest_int('num_leaves', 20, 500),
                      'max_depth': trial.suggest_int('max_depth', 4, 15),
                      # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 4),  # lambda_l1
                      # 'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 4),  # lambda_l2
                      'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)}  # feature_fraction

        # Fixed parameters

        learning_rate = 0.01
        feature_pre_filter = False
        boost_from_average = True
        n_estimators = 10000
        subsample_freq = 1
        # linear_tree = True

        params['learning_rate'] = learning_rate
        params['n_estimators'] = n_estimators
        params['feature_pre_filter'] = feature_pre_filter
        params['boost_from_average'] = boost_from_average
        params['subsample_freq'] = subsample_freq
        params['verbose'] = -1
        # params['linear_tree'] = linear_tree

        # Tweedie check

        if 'tweedie' in self.optimised_value:
            params['objective'] = trial.suggest_categorical('objective', ['tweedie'])
            params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.01, 1.91, step = 0.1)
            params['metric'] = trial.suggest_categorical('metric', ['tweedie'])  # for early_stopping

        else:
            params['objective'] = trial.suggest_categorical('objective', ['regression'])
            params['metric'] = trial.suggest_categorical('metric', ['rmse'])  # for early_stopping

        # Model

        model = lgb.LGBMRegressor(**params, n_jobs=-1)

        # Saving some parameters

        trial_results = func_to_minimise_lgb(model=model, x_train=self.x_train, y_train=self.y_train,
                                             x_val_es=self.x_val_es, y_val_es=self.y_val_es,
                                             cv=self.cv, optimised_value=self.optimised_value,
                                             suppress_warnings=self.suppress_warnings)

        trial.set_user_attr('learning_rate', learning_rate)
        trial.set_user_attr('n_estimators', trial_results[1])  # saving n_estimators after early_stopping
        trial.set_user_attr('feature_pre_filter', feature_pre_filter)
        trial.set_user_attr('boost_from_average', boost_from_average)
        trial.set_user_attr('subsample_freq', subsample_freq)
        # trial.set_user_attr('linear_tree', linear_tree)

        return trial_results[0]


class callback_optuna_lgb:
    def __init__(self, n_trials):
        self.n_trials = n_trials

    def __call__(self, study, trial):
        print(trial.number + 1, '/', self.n_trials, 'Avg score:', round(trial.value, 2))


if __name__ == '__main__':
    custom_cv()
    func_to_minimise_lgb()
    objective_lgb()
    callback_optuna_lgb()