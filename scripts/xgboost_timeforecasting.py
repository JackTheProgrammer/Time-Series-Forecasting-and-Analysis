# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from xgboost import plot_importance
from warnings import filterwarnings

filterwarnings('ignore')
plt.style.use('fivethirtyeight')
color_palette = sns.color_palette()

# %% [markdown]
# # Types of Time Series Data
# ![Time series](https://miro.medium.com/v2/resize:fit:1400/1*V_RKPeIxCB9CS_2SsLyKXw.jpeg)

# %%
df = pd.read_csv('../data/PJME_hourly.csv',index_col=[0], parse_dates=[0])
df

# %%
df.plot(style='.',
        figsize=(15, 5),
        color=color_palette[0],
        title='PJME Energy Use in MW')
plt.show()

# %% [markdown]
# # Train / Test Split

# %%
train_test_date = pd.to_datetime('01-01-2015')

train = df.loc[df.index < train_test_date]
test = df.loc[df.index >= train_test_date]

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline(train_test_date, color='black', ls='--')
ax.legend(['Training Set', 'Test Set'], loc = 'upper right', bbox_to_anchor=(1.15, 0.94))
plt.show()

# %%
plt.figure(figsize=(15,5))
sns.lineplot(
    data = df[(df.index > pd.to_datetime('01-01-2010')) & (df.index < pd.to_datetime('01-08-2010'))]
)

# %% [markdown]
# # Feature Creation

# %%
def create_features(df: pd.DataFrame):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)
df

# %% [markdown]
# ## Visualize our Feature / Target Relationship

# %%
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(data=df, x='hour', y='PJME_MW', palette='flare', hue='quarter')
ax.set_title('MW by Hour With Quarters Impact')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y='PJME_MW', palette='magma')
ax.set_title('MW by Hour')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='PJME_MW', palette='Reds')
ax.set_title('MW by Month')
plt.show()

# %% [markdown]
# # Training the model

# %%
training = create_features(train)
testing = create_features(test)

FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
TARGET = ['PJME_MW']

X_train, y_train = training[FEATURES], training[TARGET]
X_test, y_test = testing[FEATURES], testing[TARGET]

# %%
reg = xgb.XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    n_estimators=1000,
    early_stopping_rounds=50,
    objective='reg:linear',
    max_depth=3,
    learning_rate=0.01
)
reg.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=110
)

# %% [markdown]
# ## Feature importance

# %%
plot_importance(booster=reg, height=0.5)

# %%
testing['prediction'] = reg.predict(X_test)
df = df.merge(testing[['prediction']], how='left', left_index=True, right_index=True)
df

# %%
testing

# %% [markdown]
# # Forecast on Test

# %%
ax = df[['PJME_MW']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('True Data and Predictions')
plt.show()

# %%
ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'] \
    .plot(figsize=(15, 5), title='Week Of Data', style='.')
df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'] \
    .plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()

# %% [markdown]
# # Error metrics

# %%
score = np.sqrt(mean_squared_error(testing['PJME_MW'], testing['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

# %%
testing['error'] = np.abs(testing['PJME_MW'] - testing['prediction'])
testing['date'] = testing.index.date
testing.groupby(['date', 'prediction'])['error'].mean().sort_values(ascending=False)

# %%
testing

# %% [markdown]
# # Look at Worst and Best Predicted Days

# %%
testing.sort_values(by='error', ascending=True)[['dayofmonth', 'month', 'year', 'hour', 'PJME_MW', 'prediction', 'error']]

# %%
testing.info()

# %%
type_mapping = {
    'date': 'datetime64[ns]'
}

testing = testing.astype(type_mapping)
testing.info()

# %%
forecasts_errors = testing.groupby(['date', 'hour', 'quarter'])\
    .mean()[['PJME_MW','prediction','error']]
forecasts_errors

# %%
forecasts_errors.index

# %%
forecasts_errors.info()

# %%
plt.figure(figsize=(14, 6))
plt.plot(testing.index, testing['PJME_MW'], label='Actual', color='blue')
plt.plot(testing.index, testing['prediction'], label='Predicted', color='orange')
plt.fill_between(testing.index, testing['PJME_MW'], testing['prediction'], color='gray', alpha=0.3)  # Shading the error

plt.xlabel("Date")
plt.ylabel("PJME_MW")
plt.title("Time Series Plot of Actual vs Predicted PJME_MW")
plt.legend()
plt.show()

# %% [markdown]
# # Hourly Time Series Forecasting using XGBoost

# %% [markdown]
# ## Data acquiring

# %%
pjme = pd.read_csv('../data/PJME_hourly.csv', index_col=[0], parse_dates=[0])
pjme

# %%
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = pjme.plot(style='.', figsize=(15,5), color=color_pal[0], title='PJM East')

# %%
year_list = list(set([ts.year for ts in pjme.index]))
year_list.sort()
year_list

# %%
pjme_split_date = pd.to_datetime('01-01-2013')
pjme_train = pjme.loc[pjme.index <= pjme_split_date].copy()
pjme_test = pjme.loc[pjme.index > pjme_split_date]

# %%
_ = pjme_test \
    .rename(columns={'PJME_MW': 'TEST SET'}) \
    .join(pjme_train.rename(columns={'PJME_MW': 'TRAINING SET'}), how='outer') \
    .plot(figsize=(15,5), title='PJM East', style='.')

# %% [markdown]
# # Feature Engineering

# %%
from pandas.api.types import CategoricalDtype

category_types = CategoricalDtype(
    categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    ordered=True
)

def create_features(
    df: pd.DataFrame,
    label=None
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(category_types)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['date_offset'] = (df.date.dt.month*100 + df.date.dt.day - 320)%1300

    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], labels=['Spring', 'Summer', 'Fall', 'Winter'])
    X = df[['hour','dayofweek','quarter','month','year','dayofyear','dayofmonth','weekofyear','weekday','season']]
    if label:
        y = df[label]
        return X, y
    return X

X_pjme_train, y_pjme_train = create_features(pjme_train, label='PJME_MW')
X_pjme_test, y_pjme_test = create_features(pjme_test, label='PJME_MW')

# %% [markdown]
# # Create XGBoostRegressor Model

# %%
xg_reg = xgb.XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=50,
    booster='gbtree',
    objective='reg:linear',
    max_depth=3,
    enable_categorical=True,
    learning_rate=0.01
)
xg_reg.fit(
    X_pjme_train, y_pjme_train,
    eval_set=[(X_pjme_train, y_pjme_train), (X_pjme_test, y_pjme_test)],
    verbose=True
)

# %% [markdown]
# ## Feature Importance
# Feature importance is a great way to get a general idea about which features the model is relying on most to make the prediction. This is a metric that simply sums up how many times each feature is split on.
# 
# We can see that the day of year was most commonly used to split trees, while hour and year came in next. Quarter has low importance due to the fact that it could be created by different dayofyear splits.

# %%
_ = plot_importance(xg_reg, height=0.9)

# %% [markdown]
# ## Model predictions

# %%
pjme_test['MW_Prediction'] = xg_reg.predict(X_pjme_test)
pjme_all = pd.concat([X_pjme_train, y_pjme_train], sort=False)
pjme_all

# %%
_ = pjme_all[['PJME_MW','MW_Prediction']].plot(figsize=(15, 5), style=['.', '.'])

# %%
# Plot the forecast with the actual values
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax, style=['.','.'])
ax.set_xbound(lower='01-01-2015', upper='02-01-2015')
ax.set_ylim(0, 60000)
plot = plt.suptitle('January 2015 Forecast vs Actual Values')

# %%
# Plot the forecast with the actual values
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax, style=['.','.'])
ax.set_xbound(lower='01-01-2015', upper='01-08-2015')
ax.set_ylim(0, 60000)
plot = plt.suptitle('First Week of January Forecast vs Actual Values')

# %%
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax, style=['.','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='07-01-2015', upper='07-08-2015')
plot = plt.suptitle('First Week of July Forecast vs Actual Values')

# %% [markdown]
# # Error Metrics On Test Set

# %%
mean_squared_error(y_true=pjme_test['PJME_MW'], y_pred=pjme_test['MW_Prediction'])

# %%
mean_absolute_error(y_true=pjme_test['PJME_MW'], y_pred=pjme_test['MW_Prediction'])

# %%
mean_absolute_percentage_error(y_true=pjme_test['PJME_MW'], y_pred=pjme_test['MW_Prediction'])

# %% [markdown]
# # Worst and Best Predicted Days

# %%
pjme_test

# %%
from pandas.api.types import CategoricalDtype

category_types = CategoricalDtype(
    categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    ordered=True
)

def create_features(
    df: pd.DataFrame,
    label=None
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(category_types)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['date_offset'] = (df.date.dt.month*100 + df.date.dt.day - 320)%1300

    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], labels=['Spring', 'Summer', 'Fall', 'Winter'])
    X = df[['hour','dayofweek','quarter','month','year','dayofyear','dayofmonth','weekofyear','weekday','season']]
    if label:
        y = df[label]
        return X, y
    return X

full_pjme = create_features(pjme)
full_pjme['PJME_MW'] = pjme['PJME_MW']
full_pjme

# %%
full_pjme_test = full_pjme[full_pjme.index > pjme_split_date]
full_pjme_test

# %%
full_pjme_test['MW_Prediction'] = pjme_test['MW_Prediction']

# %%
full_pjme_test

# %%
full_pjme_test['error'] = full_pjme_test['PJME_MW'] - full_pjme_test['MW_Prediction']
full_pjme_test['abs_error'] = full_pjme_test['error'].apply(np.abs)
full_test_df = full_pjme_test.select_dtypes(include=np.number)
error_by_day = full_test_df.groupby(['year','month','dayofmonth']) \
    .mean()[['PJME_MW','MW_Prediction','error','abs_error']]
error_by_day

# %% [markdown]
# The best predicted days seem to be a lot of october (not many holidays and mild weather) Also early may

# %%
# Best predicted days
error_by_day.sort_values('abs_error', ascending=True).head(10)

# %% [markdown]
# # Plotting some best/worst predicted days

# %%
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax, style=['.','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='08-13-2016', upper='08-14-2016')
plot = plt.suptitle('Aug 13, 2016 - Worst Predicted Day')

# %%
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax, style=['.','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='10-03-2016', upper='10-04-2016')
plot = plt.suptitle('Oct 3, 2016 - Best Predicted Day')

# %%
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax, style=['.','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='08-13-2016', upper='08-14-2016')
plot = plt.suptitle('Aug 13, 2016 - Worst Predicted Day')

# %% [markdown]
# # Suggestions
# * Add Lag variables.
# * Add holidays.
# * Add weather data source.