# Bikes_users_count_prediction.ipynb

1. Importing necessary libraries

import numpy as np,pandas as pd, matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import matplotlib.ticker as mtick
import seaborn as sns
sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

#Importing require libraries

#Setting Format
pd.options.display.float_format = '{:.5f}'.format
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.random.seed(100)
from IPython.display import display,HTML
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import os, sys
import random

# configure font_scale and linewidth for seaborn
sns.set_context('paper', font_scale=1.3, rc={"lines.linewidth": 2})

# preprocessing and metrics
from sklearn.metrics import mean_squared_error, mean_squared_log_error, make_scorer

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# regresson model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# pipeline builder
from sklearn.pipeline import Pipeline, make_pipeline
2. Reading and Understanding the data

day = pd.read_csv("/content/drive/MyDrive/day.csv")
hour = pd.read_csv("/content/drive/MyDrive/hour.csv")

day.head()

hour.head()

day.info()

hour.info()

day.describe()

hour.describe()

day.shape

hour.shape

day.columns.to_list()

hour.columns.to_list()

day.isnull().sum()

hour.isnull().sum()

day.nunique()

hour.nunique()

day.shape,day.drop_duplicates().shape

hour.shape,hour.drop_duplicates().shape

day['temp'] = day['temp']*41
hour['temp'] = hour['temp']*41

day['atemp'] = day['atemp']*50
hour['atemp'] = hour['atemp']*50

day['hum'] = day['hum']*100
hour['hum'] = hour['hum']*100

day['windspeed'] = day['windspeed']*67
hour['windspeed'] = hour['windspeed']*67

day.head()

hour.head()

hour.rename(columns={'instant':'rec_id','dteday':'datetime','holiday':'is_holiday','workingday':'is_workingday',
                        'weathersit':'weather_condition','hum':'humidity','mnth':'month',
                        'cnt':'total_count','hr':'hour','yr':'year'},inplace=True)
hour.head()

day.rename(columns={'instant':'rec_id','dteday':'datetime','holiday':'is_holiday','workingday':'is_workingday',
                        'weathersit':'weather_condition','hum':'humidity','mnth':'month',
                        'cnt':'total_count','hr':'hour','yr':'year'},inplace=True)
day.head()

3. Cleaning data

# date column and instant cannot be used as a feature so lets drop it

day.drop(['datetime', 'rec_id'], axis = 1, inplace= True)

# we will drop holiday as workingday column fulfills the requirement efficiently.

day.drop(['is_holiday'], axis = 1, inplace = True)

day.head()

Data preprocessing

Converting Categorical Variables

season_codes = {1:'spring',2:'summer',3:'fall',4:'winter'}
day['season'] = day['season'].map(season_codes)

day.season.value_counts()

weather_condition_codes = {1:'Clear',2:'Mist',3:'Light Snow',4:'Heavy Rain'}
day['weather_condition'] = day['weather_condition'].map(weather_condition_codes)

day['weather_condition'].value_counts()

working_codes = {1:'working_day',0:'Holiday'}
day['is_workingday'] = day['is_workingday'].map(working_codes)

day['is_workingday'].value_counts()

0s
month_codes = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
day['month'] = day['month'].map(month_codes)

day['month'].value_counts()

weekday_codes = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
day['weekday'] = day['weekday'].map(weekday_codes)

day['weekday'].value_counts()

year_codes = {0:"2011",1:"2012"}
day['year'] = day['year'].map(year_codes)

day['year'].value_counts()

day.head(5)

cont_col =[ 'temp','atemp','humidity','windspeed','casual','registered']

cat_col =['season','year','month','weekday','is_workingday','weather_condition']

tg = ['total_count']

len(cont_col) + len(cat_col) + len(tg)

for i in cat_col:
    print("Name of {} cat_col".format(i)) #Name of Col
    print("No. of NUnique", day[i].nunique()) #Total Nunique Values
    print("Unique Values", day[i].unique())# All unique vales
    print('*'*30) # to make differnce i each col
    print()
    print()

def grab_col_names(dataframe, cat_th=13, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # binary cols
    binary_cols = [col for col in cat_cols if dataframe[col].nunique()==2]

    # without binary
    only_cat_cols = list(set(cat_cols)-set(binary_cols))

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'num_cols: {len(num_cols)}')
    print(f'only_cat_cols: {len(only_cat_cols)}')
    print(f'binary_cols: {len(binary_cols)}')
    return only_cat_cols, num_cols, binary_cols

only_cat_cols, num_cols, binary_cols = grab_col_names(day)

print(f'Numerical Variable: {num_cols}')
Numerical Variable: ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'total_count']

print(f'Categorical Variable: {only_cat_cols}')
Categorical Variable: ['month', 'weekday', 'weather_condition', 'season']

print(f'Binary Variable: {binary_cols}')
Binary Variable: ['year', 'is_workingday']

# Check Outliers:

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(day, col))

Dataset has no outliers.

#Summary of the dataset
day.describe()

# let's plot the distributions of the different columns
day.hist(rwidth=0.9, figsize=(20, 20))
plt.tight_layout()
plt.show()

Total count distribution is not normal, might need a change of variable

# Distribution of target variable
_ , ax = plt.subplots(1,2, figsize=(15,5)) # 1 row 2 column subplot

sns.distplot(day.total_count, bins=50, ax=ax[0]) # dependent variable distribution plot with 50 bins
ax[0].set_title('Dist plot')

ax[1] = day.total_count.plot(kind='kde') # dependent variable KDE plot
ax[1].set_title('KDE plot')

# Holiday wise yearly count of bike rental
day[['season','year', 'total_count', 'is_workingday']].groupby([ 'year', 'is_workingday']).sum()

# weather condition wise  count of bike rental
day[['weather_condition', 'total_count']].groupby(['weather_condition']).sum()

# weather condition wise  avg_count of bike rental
day[['weather_condition', 'total_count']].groupby(['weather_condition']).mean().round().astype(int)

# Days wise average causal and registered bike count

casual_avg =day.groupby(['weekday'])['casual'].mean().round().astype(int) # average casual rental on weekday
registered_avg = day.groupby(['weekday'])['registered'].mean().round().astype(int) # average casual rental on weekday

print('Total count: casual {}, registered {}'.format(casual_avg.sum(), registered_avg.sum()))

Univaraite Analysis

for col in cont_col:
    sns.displot(day[col],kde=True,edgecolor='white',height=4)
    plt.show()

Looking at the outliers, skewness and overall distribution shape using box plot

temp and atemp have similar kind of distribution, some features are showing near by normal distribution while some are skewed

for i in hour.select_dtypes(include='float'):
    sns.distplot(hour[i])
    plt.show()

for i in hour.select_dtypes(include='int'):
    sns.distplot(hour[i]) #Lets check how data is distributed
    plt.show()

for col in cont_col:
    plt.figure(figsize=(4,3))
    sns.boxplot(day[col],orient="h",width=0.8)
    plt.show()

for i in hour.select_dtypes(include='int'):
    sns.boxplot(hour[i]) #Is their any outlier
    plt.show()

Casual, Registered and Total count have outlier and need to be fix

Point to be noted that Total count is sum total of casual and Registered.

for i in hour.select_dtypes(include='float'):
    sns.boxplot(hour[i])
    plt.show()

for i in cat_col:
    sns.countplot(day[i])
    plt.show()

for i in cat_col:
    print(day[i].value_counts())
    print('~~~~~~'*10,'\n\n')

def get_df_name(df):
    '''
    This Function returns the name of a dateset
    '''
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def plot_stack_bar_chart(data, col, name):
    plt.figure(figsize=(12,8)) #Size of a PLot
    p1 = plt.bar(data[col].unique(),  # the x locations for the groups
                data.groupby([col])['casual'].sum()) # Count of casual per season

    p2 = plt.bar(data[col].unique(),  # the x locations for the groups
                data.groupby([col])['registered'].sum(), # Count of Registered per season
                 bottom = data.groupby([col])['casual'].sum()) # Count of casual per season

    plt.ylabel('Count')
    plt.title("Count by Casual and Registered for each {} in {} Data".format(col, get_df_name(data)))
    plt.xticks(data[col].unique(), name) # Name of unique values in columns
    plt.legend((p1[0], p2[0]), ('Casual', 'Registered')) #setting legends as per target
    plt.show()

# Season

fig, axs = plt.subplots(ncols=2,nrows=2, figsize=(10,5))
sns.barplot(x="season", y="casual", data=day, ci=None, ax=axs[0,0])
axs[0,0].set_title('Casual')
sns.barplot(x="season", y="registered", data=day, ci=None,ax=axs[0,1])
axs[0,1].set_title('Registered')
sns.scatterplot(x = "season", y = "casual",data=day, ax=axs[1,0])
sns.scatterplot(x = "season", y = "registered",data=day, ax=axs[1,1])


#Year

fig, axs = plt.subplots(ncols=2,nrows=2, figsize=(10,5))
sns.barplot(x="year", y="casual", data=day, ci=None,ax=axs[0,0])
axs[0,0].set_title('Casual')
sns.barplot(x="year", y="registered", data=day, ci=None,ax=axs[0,1])
axs[0,1].set_title('Registered')
sns.scatterplot(x = "year", y = "casual",data=day, ax=axs[1,0])
sns.scatterplot(x = "year", y = "registered",data=day, ax=axs[1,1])
# 1: 2012, 0: 2011

Bivariate Analysis¶

for col in cont_col:
    plt.figure(figsize=(4,3))
    sns.scatterplot(data=day,x=col,y='total_count',hue =day['is_workingday'])
    plt.show()

for col in cont_col:
    plt.figure(figsize=(4,3))
    sns.scatterplot(data=day,x=col,y='total_count',hue =day['year'],palette='magma')
    plt.show()

# Visualization of  variable varition and thier co-relation
ax = sns.pairplot(day) # pairplot
ax.fig.suptitle(' variable varition and thier co-relation',  y=1.0) # set title

fig, ax = plt.subplots(nrows=2, ncols=1)
sns.kdeplot(x='total_count', data=day, fill=True, color='#26577C', ax=ax[0])
sns.lineplot(x='weekday', y='total_count', data=day, color='#26577C', ax=ax[1])
fig.suptitle('total_count: Target Variable', y=0.95);

We have 2 years of data. We see a clear yearly seasonality and a slight positive trend.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': round(100 * (dataframe[col_name].value_counts()) / len(dataframe), 2)}))

    if plot:
        sns.countplot(x=col_name, data=dataframe)
        plt.show()

plot_stack_bar_chart(day, 'season', ('1:spring', '2:summer', '3:fall', '4:winter'))

cat_summary(day, 'season', plot=True)

plot_stack_bar_chart(day, 'year', ('2011', '2012'))

# Violin plot is used for Year-wise distribution
fig,ax=plt.subplots(figsize=(20,8))
sns.violinplot(data=day[['year','total_count']],x='year',y='total_count',ax=ax)
ax.set(title='Year-wise distribution of raidership counts')

From the bar chart, we can observed that the bike rental count distribution is highest in year 2012 then the in year 2011. Here, year 0-> 2011, year 1-> 2012

The above distribution clearly helps us to understand the multimodal distribution in both 2011 and 2012 raidership counts.The distribution for 2012 has peaks at highest values as compared with the distribution for 2011.

plot_stack_bar_chart(day, 'month', [str(i) for i in day['month'].unique()])

plot_stack_bar_chart(day, 'is_workingday', ('Yes', 'No'))

plot_stack_bar_chart(day, 'weekday', [str(i) for i in day['weekday'].unique()])

fig,ax1=plt.subplots(figsize=(15,8))
#Bar plot for weather_condition distribution of counts
sns.barplot(x='weather_condition',y='total_count',data=day[['month','total_count','weather_condition']],ax=ax1)
ax1.set_title('Weather_condition wise monthly distribution of counts')
plt.show()

From the above bar plot, we can observed that during clear,partly cloudy weather the bike rental count is highest and the second highest is during mist cloudy weather and followed by third highest during light snow weather.

plt.figure(figsize=(12,8))
sns.barplot(x = day['is_workingday'], y = day['total_count'],hue = day['season'])
plt.title('Holiday wise distribution of counts')
plt.show()

From the above bar plot, we can observed that during no holiday the bike rental counts is highest compared to during holiday for different seasons.

plt.figure(figsize=(18,10))
sns.barplot(x = day['month'], y = day['total_count'], hue = day['season'])
plt.title('Month wise distribution of counts')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to Season
sns.barplot(x='month',y='casual',data=day[['month','casual','season']],hue='season',ax=ax)
ax.set_title('Seasonal(MonthlyDistributionCasualRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to Season
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','season']],hue='season',ax=ax1)
ax1.set_title('Seasonal(MonthlyDistributionRegisteredRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to Season
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','season']],hue='season',ax=ax2)
ax2.set_title('Seasonal(MonthlyDistributionTotalRidersDaily)')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to Season
sns.barplot(x='month',y='casual',data=day[['month','casual','season']],hue='season',ax=ax)
ax.set_title('Seasonal(MonthlyDistributionCasualRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to Season
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','season']],hue='season',ax=ax1)
ax1.set_title('Seasonal(MonthlyDistributionRegisteredRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to Season
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','season']],hue='season',ax=ax2)
ax2.set_title('Seasonal(MonthlyDistributionTotalRidersHourly)')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to holiday
sns.barplot(x='month',y='casual',data=day[['month','casual','is_workingday']],hue='is_workingday',ax=ax)
ax.set_title('Holidays(MonthlyDistributionCasualRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to holiday
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','is_workingday']],hue='is_workingday',ax=ax1)
ax1.set_title('Holidays(MonthlyDistributionRegisteredRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to holiday
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','is_workingday']],hue='is_workingday',ax=ax2)
ax2.set_title('Holidays(MonthlyDistributionTotalRidersDaily)')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to holiday
sns.barplot(x='month',y='casual',data=day[['month','casual','is_workingday']],hue='is_workingday',ax=ax)
ax.set_title('Holidays(MonthlyDistributionCasualRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to holiday
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','is_workingday']],hue='is_workingday',ax=ax1)
ax1.set_title('Holidays(MonthlyDistributionRegisteredRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to holiday
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','is_workingday']],hue='is_workingday',ax=ax2)
ax2.set_title('Holidays(MonthlyDistributionTotalRidersHourly)')
plt.show()

#Month

fig, axs = plt.subplots(ncols=2,nrows=2, figsize=(10,5))
sns.barplot(x="month", y="casual", data=day, ci=None, hue='year',ax=axs[0,0])
axs[0,0].set_title('Casual')
sns.barplot(x="month", y="registered", data=day, ci=None,hue='year',ax=axs[0,1])
axs[0,1].set_title('Registered')
sns.scatterplot(x = "month", y = "casual",data=day, ax=axs[1,0])
sns.scatterplot(x = "month", y = "registered",data=day, ax=axs[1,1])

#Weekday

fig, axs = plt.subplots(ncols=2,nrows=2, figsize=(10,5))
sns.barplot(x="weekday", y="casual", data=day, ci=None,ax=axs[0,0])
axs[0,0].set_title('Casual')
sns.barplot(x="weekday", y="registered", data=day, ci=None,ax=axs[0,1])
axs[0,1].set_title('Registered')
sns.scatterplot(x = "weekday", y = "casual",data=day, ax=axs[1,0])
sns.scatterplot(x = "weekday", y = "registered",data=day, ax=axs[1,1])

#Weekday

fig, axs = plt.subplots(ncols=2,nrows=2, figsize=(10,5))
sns.barplot(x="weekday", y="casual", data=day, ci=None,ax=axs[0,0])
axs[0,0].set_title('Casual')
sns.barplot(x="weekday", y="registered", data=day, ci=None,ax=axs[0,1])
axs[0,1].set_title('Registered')
sns.scatterplot(x = "weekday", y = "casual",data=day, ax=axs[1,0])
sns.scatterplot(x = "weekday", y = "registered",data=day, ax=axs[1,1])

#Working Day

fig, axs = plt.subplots(ncols=2,nrows=2, figsize=(10,5))
sns.barplot(x="is_workingday", y="casual", data=day, ci=None,ax=axs[0,0])
axs[0,0].set_title('Casual')
sns.barplot(x="is_workingday", y="registered", data=day, ci=None,ax=axs[0,1])
axs[0,1].set_title('Registered')
sns.scatterplot(x = "is_workingday", y = "casual",data=day, ax=axs[1,0])
sns.scatterplot(x = "is_workingday", y = "registered",data=day, ax=axs[1,1])

#Weather Category

fig, axs = plt.subplots(ncols=2,nrows=2, figsize=(10,5))
sns.barplot(x="weather_condition", y="casual", data=day, ci=None,ax=axs[0,0])
axs[0,0].set_title('Casual')
sns.barplot(x="weather_condition", y="registered", data=day, ci=None,ax=axs[0,1])
axs[0,1].set_title('Registered')
sns.scatterplot(x = "weather_condition", y = "casual",data=day, ax=axs[1,0])
sns.scatterplot(x = "weather_condition", y = "registered",data=day, ax=axs[1,1])

#1:Clear, Few clouds, Partly cloudy, Partly cloudy
#2:Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#3:Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to weekday
sns.barplot(x='month',y='casual',data=day[['month','casual','weekday']],hue='weekday',ax=ax)
ax.set_title('Weekdays(MonthlyDistributionCasualRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to weekday
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','weekday']],hue='weekday',ax=ax1)
ax1.set_title('Weekdays(MonthlyDistributionRegisteredRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to weekday
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','weekday']],hue='weekday',ax=ax2)
ax2.set_title('Weekdays(MonthlyDistributionTotalRidersDaily)')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to weekday
sns.barplot(x='month',y='casual',data=day[['month','casual','weekday']],hue='weekday',ax=ax)
ax.set_title('Weekdays(MonthlyDistributionCasualRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to weekday
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','weekday']],hue='weekday',ax=ax1)
ax1.set_title('Weekdays(MonthlyDistributionRegisteredRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to weekday
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','weekday']],hue='weekday',ax=ax2)
ax2.set_title('Weekdays(MonthlyDistributionTotalRidersHourly)')
plt.show()

f, (ax1, ax2)  =  plt.subplots(nrows=1, ncols=2, figsize=(13, 6)) # 1 row 2 column subplot

# Counts of Bike Rentals by season
ax1 = day[['season','total_count']].groupby(['season']).sum().reset_index().plot(kind='bar',legend = False, title ="Counts of Bike Rentals by season", stacked=True, fontsize=12, ax=ax1)
ax1.set_xlabel("season", fontsize=12)  # set x-axis labels
ax1.set_ylabel("Count", fontsize=12)  # set y-axis labels
ax1.set_xticklabels(['spring','summer','fall','winter'])  # set x-tick labels

# Counts of Bike Rentals by weather condition
ax2 = day[['weather_condition','total_count']].groupby(['weather_condition']).sum().reset_index().plot(kind='bar', legend = False, stacked=True, title ="Counts of Bike Rentals by weather condition", fontsize=12, ax=ax2)
ax2.set_xlabel("weather_condition", fontsize=12)  # set x-axis labels
ax2.set_ylabel("Count", fontsize=12)  # set y-axis labels
ax2.set_xticklabels(['1: Clear','2: Mist','3: Light Snow'])  # set x-tick labels

f.tight_layout()

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to workingday
sns.barplot(x='month',y='casual',data=day[['month','casual','is_workingday']],hue='is_workingday',ax=ax)
ax.set_title('Workingdays(MonthlyDistributionCasualRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to workingday
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','is_workingday']],hue='is_workingday',ax=ax1)
ax1.set_title('Workingdays(MonthlyDistributionRegisteredRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to workingday
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','is_workingday']],hue='is_workingday',ax=ax2)
ax2.set_title('Workingdays(MonthlyDistributionTotalRidersDaily)')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to workingday
sns.barplot(x='month',y='casual',data=day[['month','casual','is_workingday']],hue='is_workingday',ax=ax)
ax.set_title('Workingdays(MonthlyDistributionCasualRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to workingday
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','is_workingday']],hue='is_workingday',ax=ax1)
ax1.set_title('Workingdays(MonthlyDistributionRegisteredRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to workingday
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','is_workingday']],hue='is_workingday',ax=ax2)
ax2.set_title('Workingdays(MonthlyDistributionTotalRidersHourly)')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to weatherCondition
sns.barplot(x='month',y='casual',data=day[['month','casual','weather_condition']],hue='weather_condition',ax=ax)
ax.set_title('WeatherCondition(MonthlyDistributionCasualRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to weatherCondition
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','weather_condition']],hue='weather_condition',ax=ax1)
ax1.set_title('WeatherCondition(MonthlyDistributionRegisteredRidersDaily)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to weatherCondition
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','weather_condition']],hue='weather_condition',ax=ax2)
ax2.set_title('WeatherCondition(MonthlyDistributionTotalRidersDaily)')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar-plot for monthly distribution of CasualRiders in respect to weatherCondition
sns.barplot(x='month',y='casual',data=day[['month','casual','weather_condition']],hue='weather_condition',ax=ax)
ax.set_title('WeatherCondition(MonthlyDistributionCasualRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of RegisteredRiders in respect to weatherCondition
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='registered',data=day[['month','registered','weather_condition']],hue='weather_condition',ax=ax1)
ax1.set_title('WeatherCondition(MonthlyDistributionRegisteredRidersHourly)')
plt.show()

#Bar-plot for monthly distribution of TotalRentals in a day in respect to weatherCondition
fig,ax2=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','weather_condition']],hue='weather_condition',ax=ax2)
ax2.set_title('WeatherCondition(MonthlyDistributionTotalRidersHourly)')
plt.show()

# Visualizing monthly raidershp counts across the seasons
fig,ax=plt.subplots(figsize=(20,8))
sns.pointplot(data=day[['month','total_count','season']],x='month',
             y='total_count',
             hue='season',ax=ax)
ax.set(title='Season wise montly distribution of raidership counts ')

From the above plots, we can observed that increasing the bike rental count in fall and winter season and then decreasing the bike rental count in summer and springseason.

Here, season 1 -> fall, season 2 -> winter, season 3 ->summer, season 4 -> spring

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')

#Bar plot for seasonwise monthly distribution of counts
sns.barplot(x='month',y='total_count',data=day[['month','total_count','season']],hue='season',ax=ax)
ax.set_title('Seasonwise monthly distribution of counts')
plt.show()

#Bar plot for weekday wise monthly distribution of counts
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','weekday']],hue='weekday',ax=ax1)
ax1.set_title('Weekday wise monthly distribution of counts')
plt.show()

#Barplot for Holiday distribution of counts
sns.barplot(data=day,x='is_workingday',y='total_count',hue='season')
ax.set_title('Holiday wise distribution of counts')
plt.show()

#Bar plot for workingday distribution of counts
sns.barplot(data=day,x='is_workingday',y='total_count',hue='season')
ax.set_title('Workingday wise distribution of counts')

From the above bar plot, we can observed that during no holiday the bike rental counts is highest compared to during holiday for different seasons.

From the above bar plot, we can observed that during workingday the bike rental counts is quite highest compared to during no workingday for different seasons. Here, 0-> No workingday, 1-> workingday

# Count of bike on workingdays/holidays for each season
day[['season','year', 'total_count', 'is_workingday']].groupby(['season',  'is_workingday']).sum().plot(kind='bar') # plotting bar graph
plt.title('Count of bike on workingdays/holidays for each season') # set title

plt.figure(figsize=(12,8))
sns.barplot(x = day['weather_condition'], y = day['total_count'],hue = day['season'])
plt.title('Weather Situation wise distribution of counts')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Barplot for Holiday distribution of riders with season
sns.barplot(data=day,x='is_workingday',y='casual',hue='season')
ax.set_title('workingday wise distribution of casual')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Barplot for Holiday distribution of riders with season
sns.barplot(data=day,x='is_workingday',y='registered',hue='season')
ax.set_title('workingday wise distribution of registered')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Barplot for Holiday distribution of riders with season
sns.barplot(data=day,x='is_workingday',y='total_count',hue='season')
ax.set_title('workingday wise distribution of TotalRentDay')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Barplot for Workingday distribution of riders with season
sns.barplot(data=day,x='is_workingday',y='casual',hue='season')
ax.set_title('Workingday wise distribution of casual')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Barplot for Workingday distribution of riders with season
sns.barplot(data=day,x='is_workingday',y='registered',hue='season')
ax.set_title('Workingday wise distribution of registered')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Barplot for Holiday distribution of riders with season
sns.barplot(data=day,x='is_workingday',y='total_count',hue='season')
ax.set_title('Workingday wise distribution of total_count')
plt.show()

fig, (axs1,axs2,axs3) = plt.subplots(ncols=3, figsize=(18,5)) # 1 row 3 column subplot

# Total Count of bike, season wise
sns.barplot(x='season', y='total_count', data=day, hue='weather_condition', ax=axs1, ci=None) # barplot
axs1.set_title('Total Count of bike, season wise') # set title
axs1.set_xticklabels(['spring','summer','fall','winter']) # set x-tick label
axs1.legend(labels=['Clean', 'Mist', 'Snow']) # set legend

# Casual Count of bike, season wise
sns.barplot(x='season', y='casual', data=day, hue='weather_condition', ax=axs2, ci=None) # barplot
axs2.set_title('Casual Count of bike, season wise') # set title
axs2.set_xticklabels(['spring','summer','fall','winter']) # set x-tick label
axs2.legend(labels=['Clean', 'Mist', 'Snow']) # set legend

# Registered Count of bike, season wise
sns.barplot(x='season', y='registered', data=day, hue='weather_condition', ax=axs3, ci=None) # barplot
axs3.set_title('Registered Count of bike, season wise') # set title
axs3.set_xticklabels(['spring','summer','fall','winter']) # set x-tick label
axs3.legend(labels=['Clean', 'Mist', 'Snow']) # set legend

From the above bar plot, we can observed that during clear the bike rental count is highest and the second highest is during mist cloudy weather and followed by third highest during light snow.

fig, (axs1,axs2,axs3) = plt.subplots(ncols=3, figsize=(18,5))  # 1 row 3 column subplot

# Total Count of bike for Each year
sns.barplot(x='year', y='total_count', data=day, hue='season', ax=axs1, ci=None) # barplot
axs1.set_title('Total Count of bike for Each year') # set tilte
axs1.legend(['spring','summer','fall','winter']) # set legend
axs1.set_xticklabels(['2011', '2012']) # set x-tick label

# Casual Count of bike for Each year
sns.barplot(x='year', y='casual', data=day, hue='season', ax=axs2, ci=None) # barplot
axs2.set_title('Casual Count of bike for Each year') # set title
axs2.legend(['spring','summer','fall','winter']) # set legend
axs2.set_xticklabels(['2011', '2012']) # set x-tick label

# Registered Count of bike for Each year
sns.barplot(x='year', y='registered', data=day, hue='season', ax=axs3, ci=None) # barplot
axs3.set_title('Registered Count of bike for Each year') # set title
axs3.legend(['spring','summer','fall','winter']) # set legend
axs3.set_xticklabels(['2011', '2012']) # set x-tick label

fig,ax=plt.subplots(figsize=(20,8))
sns.pointplot(x='weekday',y='total_count',data=day[['total_count','weekday']],hue='weekday')
ax.set_title('Weekday wise hourly distribution of counts')
plt.show()

fig,ax1=plt.subplots(figsize=(20,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count']],ax=ax1)
ax1.set_title('Monthly distribution of counts')
plt.show()
fig,ax2=plt.subplots(figsize=(20,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','season']],hue='season',ax=ax2)
ax2.set_title('Season wise monthly distribution of counts')
plt.show()

fig,ax1=plt.subplots(figsize=(20,8))
sns.barplot(x='weekday',y='total_count',data=day[['weekday','total_count']],ax=ax1)
ax1.set_title('Monthly distribution of counts')
plt.show()
fig,ax2=plt.subplots(figsize=(20,8))
sns.barplot(x='weekday',y='total_count',data=day[['weekday','total_count','season']],hue='season',ax=ax2)
ax2.set_title('Season wise monthly distribution of counts')
plt.show()

fig,ax=plt.subplots(figsize=(20,8))
sns.violinplot(x='year',y='total_count',data=day[['year','total_count']])
ax.set_title('Yearly wise distribution of counts')
plt.show()
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(20,5))
sns.barplot(data=day,x='is_workingday',y='total_count',hue='season',ax=ax1)
ax1.set_title('is_workingday wise distribution of counts')
sns.barplot(data=day,x='is_workingday',y='total_count',hue='season',ax=ax2)
ax2.set_title('is_workingday wise distribution of counts')
plt.show()

#Consistency of Bike count on weekdays by monthly basis
plt.figure(figsize=(18,8)) # set figure size

sns.pointplot(x='month', y='total_count', data=day, hue='weekday', ci=None) # pointplot
plt.legend(['Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday', 'Sunday']) # set legend
plt.title('Bike count on weekdays by monthly basis') # set title

 ax=plt.subplots(nrows=2, ncols=1, figsize=(15,10))

# Total count: season wise of weekdays
sns.pointplot(x='season', y='total_count', data=day, hue='weekday', ci=None, ax=ax[0]) # pointplot
ax[0].set_title('Total count: season wise of weekdays') # set title
ax[0].legend(['Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday', 'Sunday']) # set legend
ax[0].set_xticklabels(['spring','summer','fall','winter']) # set x-tick label

# Total count: season wise of both year
sns.pointplot(x='season', y='total_count', data=day, hue='year', ci=None, ax=ax[1]) # pointplot
ax[1].set_title('Total count: season wise of both year') # set title
ax[1].legend(['2011', '2012']) # set legend
ax[1].set_xticklabels(['spring','summer','fall','winter']) # set x-tick label

# Avg Use of the bikes by casual users on weekdays
_, ax=plt.subplots(nrows=2, ncols=1, figsize=(15,10)) # 2 row 1 col subplot
sns.pointplot(x='weekday', y='casual', data=day, ci=None, ax=ax[0]) # pointplot
ax[0].set(title="Avg Use of the bikes by casual users") # set title
ax[0].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday', 'Sunday']) # set x-tick label

for c in ax[0].collections:
    for val,of in zip(casual_avg,c.get_offsets()):
        ax[0].annotate(val, of)                         # set annotations for average of each weekday for  casual rentals

# Avg Use of the bikes by registered users on weekday
sns.pointplot(x='weekday', y='registered', data=day, ci=None, ax=ax[1]) # set pointplot
ax[1].set(title="Avg Use of the bikes by registered users") # set title
ax[1].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday', 'Sunday']) # set x-tick label

for c in ax[1].collections:
    for val,of in zip(registered_avg,c.get_offsets()):
        ax[1].annotate(val, of)                          # set annotations for average of each weekday for registered rentals

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar plot for seasonwise monthly distribution of counts
sns.barplot(x='month',y='total_count',data=day[['month','total_count','season']],hue='season',ax=ax)
ax.set_title('Seasonwise monthly distribution of counts')
plt.show()
#Bar plot for weekday wise monthly distribution of counts
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=day[['month','total_count','weekday']],hue='weekday',ax=ax1)
ax1.set_title('Weekday wise monthly distribution of counts')
plt.show()

From the above plots, we can observed that increasing the bike rental count in spring and summer season and then decreasing the bike rental count in fall and winter season. Here,

season 1-> summer season 2 -> spring season 3 -> fall season 4 -> winter

fig, ax = plt.subplots(figsize=(20,5))
sns.barplot(data=day, x='month', y='total_count', ax=ax)
ax.set(title='Count of bikes during different months')

The above distribution shows highest raidership counts for the month June-September & lowest count for January month.

fig, ax = plt.subplots(figsize=(20,5))
sns.barplot(data=day, x='weekday', y='total_count', ax=ax)
ax.set(title='Count of bikes during different days')

Observation: all Categorical features people like to rent bikes more when the sky is clear. the count of number of rented bikes is maximum in summer season and least in winter season. number of bikes rented per season over the years has increased for both casual and registered users. registered users have rented more bikes than casual users overall. casual users travel more over weekends as compared to registered users (Saturday / Sunday). registered users rent more bikes during working days as expected for commute to work / office. demand for bikes are more on working days as compared to holidays ( because majority of the bike users are registered )

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders
sns.violinplot(x='year',y='casual',data=day[['year','casual']])
ax.set_title('yearly distribution of CasualRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders
sns.violinplot(x='year',y='registered',data=day[['year','registered']])
ax.set_title('yearly distribution of RegisteredRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders
sns.violinplot(x='year',y='total_count',data=day[['year','total_count']])
ax.set_title('yearly distribution of TotalRidersDays')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders
sns.violinplot(x='year',y='casual',data=day[['year','casual']])
ax.set_title('Yearly distribution of CasualRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders
sns.violinplot(x='year',y='registered',data=day[['year','registered']])
ax.set_title('Yearly distribution of RegisteredRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders
sns.violinplot(x='year',y='total_count',data=day[['year','total_count']])
ax.set_title('Yearly distribution of TotalRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders
sns.violinplot(x='season',y='casual',data=day[['season','casual']])
ax.set_title('Seasonal distribution of CasualRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders
sns.violinplot(x='season',y='registered',data=day[['season','registered']])
ax.set_title('Seasonal distribution of RegisteredRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders
sns.violinplot(x='season',y='total_count',data=day[['season','total_count']])
ax.set_title('Seasonal distribution of TotalRidersDays')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders
sns.violinplot(x='season',y='casual',data=day[['season','casual']])
ax.set_title('Seasonal distribution of CasualRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders
sns.violinplot(x='season',y='registered',data=day[['season','registered']])
ax.set_title('Seasonal distribution of RegisteredHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders
sns.violinplot(x='season',y='total_count',data=day[['season','total_count']])
ax.set_title('Seasonal distribution of TotalRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders
sns.violinplot(x='is_workingday',y='casual',data=day[['is_workingday','casual']])
ax.set_title('Holiday distribution of CasualRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders
sns.violinplot(x='is_workingday',y='registered',data=day[['is_workingday','registered']])
ax.set_title('Holiday distribution of RegisteredRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders
sns.violinplot(x='is_workingday',y='total_count',data=day[['is_workingday','total_count']])
ax.set_title('Holiday distribution of TotalRidersDays')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders
sns.violinplot(x='is_workingday',y='casual',data=day[['is_workingday','casual']])
ax.set_title('Holiday distribution of CasualRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders
sns.violinplot(x='is_workingday',y='registered',data=day[['is_workingday','registered']])
ax.set_title('Holiday distribution of RegisteredRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders
sns.violinplot(x='is_workingday',y='total_count',data=day[['is_workingday','total_count']])
ax.set_title('Holiday distribution of TotalRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders
sns.violinplot(x='weekday',y='casual',data=day[['weekday','casual']])
ax.set_title('Weekday distribution of CasualRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders
sns.violinplot(x='weekday',y='registered',data=day[['weekday','registered']])
ax.set_title('Weekday distribution of RegisteredRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders
sns.violinplot(x='weekday',y='total_count',data=day[['weekday','total_count']])
ax.set_title('Weekday distribution of TotalRidersDays')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders
sns.violinplot(x='weekday',y='casual',data=day[['weekday','casual']])
ax.set_title('Weekday distribution of CasualRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders
sns.violinplot(x='weekday',y='registered',data=day[['weekday','registered']])
ax.set_title('Weekday distribution of RegisteredRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders
sns.violinplot(x='weekday',y='total_count',data=day[['weekday','total_count']])
ax.set_title('Weekday distribution of TotalRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for workingdays distribution of riders
sns.violinplot(x='is_workingday',y='casual',data=day[['is_workingday','casual']])
ax.set_title('Workingday distribution of CasualRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for workingdays distribution of riders
sns.violinplot(x='is_workingday',y='registered',data=day[['is_workingday','registered']])
ax.set_title('Workingday distribution of RegisteredRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders
sns.violinplot(x='is_workingday',y='total_count',data=day[['is_workingday','total_count']])
ax.set_title('Workingday distribution of TotalRidersDays')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for workingdays distribution of riders
sns.violinplot(x='is_workingday',y='casual',data=day[['is_workingday','casual']])
ax.set_title('Workingday distribution of CasualRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for workingdays distribution of riders
sns.violinplot(x='is_workingday',y='registered',data=day[['is_workingday','registered']])
ax.set_title('Workingday distribution of RegisteredRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Violin plot for weekdays distribution of riders
sns.violinplot(x='is_workingday',y='total_count',data=day[['is_workingday','total_count']])
ax.set_title('Workingday distribution of TotalRidersHourly')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders
sns.violinplot(x='weather_condition',y='casual',data=day[['weather_condition','casual']])
ax.set_title('WeatherCondition distribution of CasualRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders
sns.violinplot(x='weather_condition',y='registered',data=day[['weather_condition','registered']])
ax.set_title('WeatherCondition distribution of RegisteredRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Violin plot for WeatherCondition distribution of riders
sns.violinplot(x='weather_condition',y='total_count',data=day[['weather_condition','total_count']])
ax.set_title('WeatherCondition distribution of TotalRidersDays')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders
sns.violinplot(x='weather_condition',y='casual',data=day[['weather_condition','casual']])
ax.set_title('WeatherCondition distribution of CasualRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders
sns.violinplot(x='weather_condition',y='registered',data=day[['weather_condition','registered']])
ax.set_title('WeatherCondition distribution of RegisteredRiders')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders
sns.violinplot(x='weather_condition',y='total_count',data=day[['weather_condition','total_count']])
ax.set_title('WeatherCondition distribution of TotalRidersHourly')
plt.show()

Finding Outliers in the dataset

sns.boxplot(data=day, x="total_count")

Most of the data is contained between 0 to ~6.

sns.boxplot(day['humidity'])

sns.boxplot(day['windspeed'])

day.describe(include='all').T

def treat_outlier_iqr(data, col):

    #Finding 25 and 75 Quantile
    q25, q75 = np.percentile(data[col], 25), np.percentile(data[col], 75)
    # Inter Quantile Range
    iqr = q75-q25
    #Minimum and Maximum Range
    min_r, max_r = q25-(iqr*1.5), q75+(iqr*1.5)
    #Replacing Outliers with Mean
    data.loc[data.loc[:, col] < min_r, col] = data[col].mean()
    data.loc[data.loc[:, col] > max_r, col] = data[col].mean()

    return sns.boxplot(data[col])

treat_outlier_iqr(day, 'humidity') # Treating Outliers in Hum Column

treat_outlier_iqr(day, 'windspeed') # Treating Outliers in Hum Column

Inference = Bike Rentals increased significantly in the year 2012 ,are more during the spring season, during clear weather and from june to september months and on holidays

fig,ax=plt.subplots(figsize=(20,8))

sns.boxplot(data=day[['temp','windspeed','humidity']])
ax.set_title('temp_windspeed_humidity distribution')
plt.show()

fig,(ax1,ax2,ax3)=plt.subplots(nrows=3,figsize=(20,10))
sns.boxplot(x='weekday',y='total_count',data=day[['weekday','total_count']],ax=ax1)
ax1.set_title('Day wise distribution of outliers')

sns.barplot(x='month',y='total_count',data=day[['month','total_count']],ax=ax2)
ax2.set_title('Monthly wise distribution of outliers')

sns.violinplot(x='year',y='total_count',data=day[['year','total_count']],ax=ax3)
ax3.set_title('Yearly wise distribution of outliers')
plt.show()

fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(20,8))
sns.boxplot(data=day[['total_count','casual','registered']],ax=ax1)
sns.boxplot(data=day[['temp','windspeed','humidity']],ax=ax2)

In above plots, the casual, windspeed, & humidity data shows the outliers.

treat_outlier_iqr(day, 'casual')

for col in cont_col:
    plt.figure(figsize=(4,3))
    sns.boxplot(day[col],orient="h",width=0.8)
    plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Boxplot for casuals outliers
sns.boxplot(data=day[['casual']])
ax.set_title('casual outliers')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Boxplot for registered outliers
sns.boxplot(data=day[['registered']])
ax.set_title('registered outliers')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Boxplot for TotalRiders outliers
sns.boxplot(data=day[['total_count']])
ax.set_title('total_count')
plt.show()

#create dataframe for outliers
CorrectionOutliers = pd.DataFrame(day,columns=['windspeed','humidity'])
#ColNames for outliers
ColNames=['windspeed','humidity']

for Col in ColNames:
    q75,q25 = np.percentile(CorrectionOutliers.loc[:,Col],[75,25]) # Divide data into 75%quantile and 25%quantile.
    iqr = q75-q25 #Inter quantile range
    min = q25-(iqr*1.5) #inner fence
    max = q75+(iqr*1.5) #outer fence
    CorrectionOutliers.loc[CorrectionOutliers.loc[:,Col] < min,:Col] = np.nan  #Replace with NA
    CorrectionOutliers.loc[CorrectionOutliers.loc[:,Col] > max,:Col] = np.nan  #Replace with NA
    
#Replacing the outliers by mean values
CorrectionOutliers['windspeed'] = CorrectionOutliers['windspeed'].fillna(CorrectionOutliers['windspeed'].mean())
CorrectionOutliers['humidity'] = CorrectionOutliers['humidity'].fillna(CorrectionOutliers['humidity'].mean())

#create dataframe for outliers
CorrectionOutlierss = pd.DataFrame(hour,columns=['windspeed','humidity'])
#ColNames for outliers
ColNamess=['windspeed','humidity']

for Col in ColNamess:
    q75,q25 = np.percentile(CorrectionOutlierss.loc[:,Col],[75,25]) # Divide data into 75%quantile and 25%quantile.
    iqr = q75-q25 #Inter quantile range
    min = q25-(iqr*1.5) #inner fence
    max = q75+(iqr*1.5) #outer fence
    CorrectionOutlierss.loc[CorrectionOutlierss.loc[:,Col] < min,:Col] = np.nan  #Replace with NA
    CorrectionOutlierss.loc[CorrectionOutlierss.loc[:,Col] > max,:Col] = np.nan  #Replace with NA
    
#Replacing the outliers by mean values
CorrectionOutlierss['windspeed'] = CorrectionOutlierss['windspeed'].fillna(CorrectionOutlierss['windspeed'].mean())
CorrectionOutlierss['humidity'] = CorrectionOutlierss['humidity'].fillna(CorrectionOutlierss['humidity'].mean())

#Replacing the WindspeedOutliers
day['windspeed']=day['windspeed'].replace(CorrectionOutliers['windspeed'])
#Replacing the HumidityOutliers
day['humidity']=day['humidity'].replace(CorrectionOutliers['humidity'])
day.head(5)

#Replacing the WindspeedOutliers
hour['windspeed']=hour['windspeed'].replace(CorrectionOutlierss['windspeed'])
#Replacing the HumidityOutliers
hour['humidity']=hour['humidity'].replace(CorrectionOutlierss['humidity'])
hour.head(5)

fig,ax=plt.subplots(figsize=(15,8))
#Boxplot for casuals outliers
sns.boxplot(data=day[['temp','atemp','windspeed','humidity']])
ax.set_title('temp_FeelingTemperature_windspeed_humidity_outliers')
plt.show()

fig,ax=plt.subplots(figsize=(15,8))
#Boxplot for casuals outliers
sns.boxplot(data=hour[['temp','atemp','windspeed','humidity']])
ax.set_title('temp_FeelingTemperature_windspeed_humidity_outliers')
plt.show()

# Outlier Analysis
fig, axes = plt.subplots(nrows=4,ncols=2) # 4 row 2 column subplots
fig.set_size_inches(20, 16) # set figure size
plt.subplots_adjust(hspace=0.3) # set hspace to avoid overlapping

# boxplots for categorical and continuous features
sns.boxplot(data=day,y="total_count", ax=axes[0][0])
sns.boxplot(data=day,y="total_count",x="season", ax=axes[0][1])
sns.boxplot(data=day,y="total_count",x="year", ax=axes[1][0])
sns.boxplot(data=day,y="total_count",x="weather_condition", ax=axes[1][1])
sns.boxplot(data=day,y="total_count",x="month", ax=axes[2][0])
sns.boxplot(data=day,y="total_count",x="weekday", ax=axes[2][1])
sns.boxplot(data=day,x="windspeed", ax=axes[3][0])
sns.boxplot(data=day,x="humidity", ax=axes[3][1])

axes[0][0].set(ylabel='Count',title="Box Plot On Count")
axes[0][1].set(ylabel='Count',title="Box Plot On Count Across Season")
axes[0][1].set_xticklabels(['spring','summer','fall','winter'])

axes[1][0].set(ylabel='Count',title="Box Plot On Count Across year")
axes[1][0].set_xticklabels(['2011','2012'])

axes[1][1].set(ylabel='Count',title="Box Plot On Count Across weather")
axes[1][1].set_xticklabels(['Clean','Mist','Snow'])

axes[2][0].set(ylabel='Count',title="Box Plot On Count Across month")
axes[2][1].set(ylabel='Count',title="Box Plot On Count Across weekday")
axes[2][1].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday', 'Sunday'])

axes[3][0].set(ylabel='windspeed',title="Box Plot On windspeed")
axes[3][1].set(ylabel='humidity',title="Box Plot On humidity")

# finding outliers
wind_humidity = pd.DataFrame(day, columns=['windspeed', 'humidity'])

# get outliers for windspeed and humidity features
for i in ['windspeed', 'humidity']:
    q75, q25 = np.percentile(wind_humidity.loc[:,i], [75,25]) # get q75 and q25
    IQR = q75 - q25 # calculate  IQR for boxplot outlier method
    max = q75+(IQR*1.5) # get max bound
    min = q25-(IQR*1.5) # get min bound
    wind_humidity.loc[wind_humidity.loc[:,i]<min,:i] = np.nan # replacing outliers with NAN
    wind_humidity.loc[wind_humidity.loc[:,i]>max,:i] = np.nan # replacing outliers with NAN

print('Shape after dropping outlier (windspeed,humidity):',wind_humidity.dropna().shape)
print('Shape before dropping outlier (windspeed,humidity):',day[['windspeed','humidity']].shape)

# calculating outlier indexs
index=[]
outlier = pd.DataFrame()
for i in range(wind_humidity.shape[0]):
    if wind_humidity.loc[i,].isna().any(): # if either of windspeed or humidity is NAN, for each column
        outlier.loc[i,'outlier'] = 1 # store index as outlier 1
        index.append(i) # store indices of outliers
    else:
        outlier.loc[i,'outlier'] = 0

wind_humidity['outlier'] = outlier['outlier'].astype(int) # convert outlier column as integer
wind_humidity.loc[index,] # show outliers with thier respective indices

day['outlier'] = wind_humidity['outlier'] # add oulier feature in bike data

#dropping all the outliers present in dataframe
day.drop(day[(day.outlier==1) ].index, inplace=True) # dropping all the outliers
print('Shape after dropping outlier:',day.shape) # shape of the after removing outlier rows
print(day.info())

import matplotlib.pyplot as plt
plt.hist(day['temp'], bins=30)
plt.xlabel('temperature(°C)')
plt.ylabel('fraction of temperature')
plt.show()

plt.hist(day['atemp'], bins=30)
plt.xlabel('atemp(°C)')
plt.ylabel('fraction of temperature')
plt.show()

plt.hist(day['humidity'], bins=30)
plt.xlabel('humidity')
plt.ylabel('fraction of humidity')
plt.show()

import matplotlib.pyplot as plt
plt.hist(day['windspeed'], bins=30)
plt.xlabel('windSpeed')
plt.ylabel('fraction of windspeed')
plt.show()

plt.hist(day['casual'], bins=30)
plt.xlabel('casual')
plt.ylabel('fraction of casual')

plt.hist(day['registered'], bins=30)
plt.xlabel('registered')
plt.ylabel('fraction of registered')
plt.show()

plt.hist(day['total_count'], bins=30)
plt.xlabel('total_count')
plt.ylabel('fraction of TotalRentDay')
plt.show()

import matplotlib.pyplot as plt
plt.hist(day['temp'], bins=30)
plt.xlabel('temperature(°C)')
plt.ylabel('fraction of temperature')
plt.show()

plt.hist(day['atemp'], bins=30)
plt.xlabel('FeelingTemp(°C)')
plt.ylabel('fraction of temperature')
plt.show()

plt.hist(day['humidity'], bins=30)
plt.xlabel('Humidity')
plt.ylabel('fraction of humidity')
plt.show()

plt.hist(day['windspeed'], bins=30)
plt.xlabel('WindSpeed')
plt.ylabel('fraction of windspeed')
plt.show()

plt.hist(day['casual'], bins=30)
plt.xlabel('casual')
plt.ylabel('fraction of casual')
plt.show()

plt.hist(day['registered'], bins=30)
plt.xlabel('registered')
plt.ylabel('fraction of registered')
plt.show()

plt.hist(day['total_count'], bins=30)
plt.xlabel('total_count')
plt.ylabel('fraction of TotalRentHourly')
plt.show()

def Pmf(data):
    return data.value_counts().sort_index()/len(data)

BusyDay = day['is_workingday'] == 1
SumRider = day['total_count']
BusyDay_SumRider = SumRider[BusyDay]
IdleDay_SumRider = SumRider[~BusyDay]
Pmf(BusyDay_SumRider).plot(label='is_workingday')
Pmf(IdleDay_SumRider).plot(label='is_workingday')
plt.xlabel('SumRider(cnt)')
plt.ylabel('Count')

# Visualization of continuous variable varition and thier co-relation
ax = sns.pairplot(day[['temp', 'atemp', 'windspeed', 'humidity']] ) # pairplot
ax.fig.suptitle('Continuous variable varition and thier co-relation',  y=1.0) # set title

# Regresson plots between temp, windspeed and humidity against total_count
_ , ax = plt.subplots(1,3, figsize=(18,5)) # 1 row 3 column subplot

sns.regplot(x = 'temp', y='total_count', data=day, ax= ax[0]) # Regression plot
ax[0].set_title('+ve relation between temp and total_count') # set title

sns.regplot(x = 'windspeed', y='total_count', data=day, ax= ax[1]) # Regression plot
ax[1].set_title('-ve relation between windspeed and total_count') # set title

sns.regplot(x = 'humidity', y='total_count', data=day, ax= ax[2]) # Regression plot
ax[2].set_title('+ve relation between humidity and total_count') # set title

# Regresson plots between temp, windspeed and humidity to show thier relation with each other
_ , ax = plt.subplots(1,3, figsize=(18,5)) # 1 row 3 column subplots

sns.regplot(x = 'temp', y='humidity', data=day, ax= ax[0])# Regression plot
ax[0].set_title('+ve relation between temp and humidity') # set title

sns.regplot(x = 'windspeed', y='humidity', data=day, ax= ax[1])# Regression plot
ax[1].set_title('-ve relation between windspeed and humidity') # set title

sns.regplot(x = 'temp', y='windspeed', data=day, ax= ax[2])# Regression plot
ax[2].set_title('+ve relation between temp and windspeed') # set title

Observation A +ve correlation between humidity and temperature was observed (as temp increases the amount of water vapour present in the air also increases) A -ve correlation between windspeed with humidity and temperature was observed (as wind increases, it draws heat from the body, thereby temperature and humidity decreases)

fig, axs = plt.subplots(ncols=2, figsize=(10,5))
sns.scatterplot(x = "temp", y = "casual",data=day, ax=axs[0])
sns.scatterplot(x = "temp", y = "registered",data=day, ax=axs[1])

fig, axs = plt.subplots(ncols=2, figsize=(10,5))
sns.scatterplot(x = "atemp", y = "casual",data=day, ax=axs[0])
sns.scatterplot(x = "atemp", y = "registered",data=day, ax=axs[1])

fig, axs = plt.subplots(ncols=2, figsize=(10,5))
sns.scatterplot(x = "atemp", y = "casual",data=day, ax=axs[0])
sns.scatterplot(x = "atemp", y = "registered",data=day, ax=axs[1])

fig, axs = plt.subplots(ncols=2, figsize=(10,5))
sns.scatterplot(x = "windspeed", y = "casual",data=day, ax=axs[0])
sns.scatterplot(x = "windspeed", y = "registered",data=day, ax=axs[1])

sns.scatterplot(x = "casual", y = "registered",data=day)

plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.title("Demand = f(Temperature)")
plt.scatter(x=day.temp, y=day.total_count, s=2, c="magenta")

##
plt.subplot(2, 2, 2)
plt.title("Demand = f(Feeled Temperature)")
plt.scatter(x=day.atemp, y=day.total_count, s=2, c="blue")

##
plt.subplot(2, 2, 3)
plt.title("Demand = f(Humidity)")
plt.scatter(x=day.humidity, y=day.total_count, s=2, c="green")

##
plt.subplot(2, 2, 4)
plt.title("Demand = f(Wind speed)")
plt.scatter(x=day.windspeed, y=day.total_count, s=2, c="red")

plt.tight_layout()

We can spot some dependency for all of these features, except maybe for humidity.

cname = [x for x in day.columns if x not in ['registered','casual','total_count','season','year','month','is_workingday','weekday','weather_condition']]

#Outlier Analysis, box plot ----------> Remove outlier
for col in cname:
    q75,q25 = np.percentile(day.loc[:,col],[75,25]);
    iqr = q75-q25;
    lower_fence = q25 - (1.5*iqr)
    upper_fence = q75 + (1.5*iqr)

    day_outlier_del = day.drop(day[day.loc[:,col] < lower_fence].index)
    day_outlier_del = day.drop(day[day.loc[:,col] > upper_fence].index)

#Outlier Analysis by considering outliers as NA and imputing values
day_outlier_impute = day.copy()
for col in cname:
    q75,q25 = np.percentile(day.loc[:,col],[75,25]);
    iqr = q75-q25;
    lower_fence = q25 - (1.5*iqr)
    upper_fence = q75 + (1.5*iqr)

    row = day_outlier_impute[day.loc[:,col] < lower_fence].index
    day_outlier_impute.loc[row,col] = np.nan

    row = day_outlier_impute[day.loc[:,col] > upper_fence].index
    day_outlier_impute.loc[row,col] = np.nan

day_outlier_impute.isna().sum()

print('Percentage of Missing value:',(15/731)*100)

day

import scipy
from scipy import stats
#Normal plot
fig=plt.figure(figsize=(15,8))
stats.probplot(day.total_count.tolist(),dist='norm',plot=plt)
plt.show()

The above probability plot, the some target variable data points are deviates from normality.

import scipy
from scipy import stats
#Normal plot
fig=plt.figure(figsize=(15,8))
stats.probplot(day.casual.tolist(),dist='norm',plot=plt)
plt.xlabel("Normality", labelpad=30)
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()

fig=plt.figure(figsize=(15,8))
stats.probplot(day.registered.tolist(),dist='norm',plot=plt)
plt.xlabel("Normality", labelpad=30)
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()

fig=plt.figure(figsize=(15,8))
stats.probplot(day.total_count.tolist(),dist='norm',plot=plt)
plt.xlabel("Normality", labelpad=30)
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()

import scipy
from scipy import stats
#Normal plot
fig=plt.figure(figsize=(15,8))
stats.probplot(hour.casual.tolist(),dist='norm',plot=plt)
plt.xlabel("Normality", labelpad=30)
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()

fig=plt.figure(figsize=(15,8))
stats.probplot(hour.registered.tolist(),dist='norm',plot=plt)
plt.xlabel("Normality", labelpad=30)
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()

fig=plt.figure(figsize=(15,8))
stats.probplot(hour.total_count.tolist(),dist='norm',plot=plt)
plt.xlabel("Normality", labelpad=30)
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()

fig=plt.figure(figsize=(15,8))
stats.probplot(hour.windspeed.tolist(),dist='norm',plot=plt)
plt.xlabel("Normality", labelpad=30)
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()

fig=plt.figure(figsize=(15,8))
stats.probplot(hour.humidity.tolist(),dist='norm',plot=plt)
plt.xlabel("Normality", labelpad=30)
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()

fig=plt.figure(figsize=(15,8))
stats.probplot(hour.atemp.tolist(),dist='norm',plot=plt)
plt.xlabel("Normality", labelpad=30)
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()

Plotting heatmap

# correlation degree of all the numerical features wrt to the total count of bike.
day[["temp", "atemp", "humidity", "windspeed", "total_count"]].corr()["total_count"].plot(kind="bar", title="Correlation of variable features wrt to total number of bikes")

windspeed is maybe less related to how much bikes are used. Let's keep that in mind.

day

day1=day.drop(['season','month','weekday','is_workingday','weather_condition'],axis=1)

plt.figure(figsize = (14,8))
sns.heatmap(day1.corr(), annot = True, cmap = 'Blues')
plt.show()

temp with atemp & count with registered/casual are highly co-realted , so we will drop them

hour

hour1=hour.drop(['datetime'],axis=1)

plt.figure(figsize = (14,8))
sns.heatmap(hour1.corr(), annot = True, cmap = 'Blues')
plt.show()

day1.corr()['total_count'] #Co-relation with Tagret Variable

# There is a high positively correlation between "temp" variable and "atemp" variable.

day = day.drop("atemp", axis=1)

# I'm going to ignore "casual" and "registered" variable for further analysis.
day = day.drop(["casual","registered"], axis=1)

def feature_eng(data, col):
    day['weekday'] = day['weekday'].astype('category') #Converting day to category
    day.drop('temp', axis=1, inplace=True) #Droping all the irrelevant column

#Transforming Hour Data
feature_eng(day, 'registered')

day.info()

day.head()

day.describe(include='all').T

# Getting all category columns to list
cat_col = day.select_dtypes(include='category').columns.to_list()

#Printing all unique values in Category Columnns
for i in cat_col:
    print("Name of {} col".format(i))
    print("No. of NUnique", hour[i].nunique())
    print("Unique Values", hour[i].unique())
    print('*'*30)

day.info()

Features Engineering of the Data

dropped = ["windspeed","is_workingday", "weekday", "year"]
day_final = day.drop(dropped, axis=1)

# categorising features
categorical_features = ("season","is_workingday","weather_condition","weekday","month","year")
continous_features = ("humidity","windspeed")
dropFeatures = ('casual',"datetime","instant","registered","atemp")
target=('total_count')

day_final.head()

# plotting pairplot
sns.pairplot(day)
plt.show()

# Let's check autocorrelation of cnt values
plt.acorr(day_final["total_count"].astype(float), maxlags=12)

Dummy variable creation

# one hot encoding day_final
day_final_encoded = pd.get_dummies(day_final, columns=['weather_condition', 'season', 'month'], dtype=int, )

hour_encoded = hour.drop('datetime',axis=1)

day_final_encoded.sample()

Splitting the Data into Training and Testing Sets

#for Spliting Data and Hyperparameter Tuning
from sklearn.model_selection import train_test_split, GridSearchCV

#Importing Machine Learning Model

from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

#statistical Tools
from sklearn import metrics

#To tranform data
from sklearn import preprocessing

#load the required libraries
from sklearn import preprocessing,metrics,linear_model
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split

X = day_final_encoded.drop("total_count", axis=1)
y = day_final_encoded["total_count"]

X_train_day, X_val_day, y_train_day, y_val_day = train_test_split(X, y, test_size=0.2, random_state=42)

# splitting hour dataset 
X_hour = hour_encoded.drop("total_count", axis=1)
y_hour = hour_encoded["total_count"]

X_train_hour, X_val_hour, y_train_hour, y_val_hour = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_day, X_val_day, y_train_day, y_val_day = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train_day.shape, X_val_day.shape, y_train_day.shape, y_val_day.shape)

X_train_hour, X_val_hour, y_train_hour, y_val_hour = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train_hour.shape, X_val_hour.shape, y_train_hour.shape, y_val_hour.shape)


Classic ML

def tune_and_evaluate(model, param_grid, X, y):
    
    """This function tunes the model using grid search to find the best 
    hyperparameters and evaluates it.
    
    Paramters: 
        model: the model that we want to tune.
        param_grid: the hyperparameter grid for grid search.
        X: the feature matrix
        y: the target variable
    
    Returns:
        best_model: the best model found during the hyperparameter search.
    """
    # initialize the grid search object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=5)
    
    # fit the object to the given data
    grid_search.fit(X, y)
    
    # extract the best parameters, score, and model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    # print the best parameters and score
    print("Best parameters are: ",best_params)
    print('Mean cross-validated score of the best estimator is: ',best_score)
    print('-------------------------------------------------------------------')
    
    return best_model

from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam, Adamax, SGD

import warnings
warnings.filterwarnings('ignore')
models_names = ['Linear Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost']

lin_reg = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

models = [
    lin_reg, KNeighborsRegressor(),
    DecisionTreeRegressor(random_state=42),
    RandomForestRegressor(random_state=42),
    XGBRegressor(random_state=42)
]

models_list = list(zip(models_names, models))

train_acc_day = {}
val_acc_day = {}

for name, model in models_list:
    model.fit(X_train_day, y_train_day)
    train_acc_day[name] = round(model.score(X_train_day, y_train_day), 2)
    val_acc_day[name] = round(model.score(X_val_day, y_val_day), 2)

train_acc_day , val_acc_day

train_acc_hour = {}
val_acc_hour = {}

for name, model in models_list:
    model.fit(X_train_hour, y_train_hour)
    train_acc_hour[name] = round(model.score(X_train_hour, y_train_hour), 2)
    val_acc_hour[name] = round(model.score(X_val_hour, y_val_hour), 2)

train_acc_hour , val_acc_hour
({'Linear Regression': 0.57,
  'KNN': 0.6,
  'Decision Tree': 1.0,
  'Random Forest': 0.91,
  'XGBoost': 0.97},
 {'Linear Regression': 0.5,
  'KNN': 0.35,
  'Decision Tree': 0.1,
  'Random Forest': 0.38,
  'XGBoost': 0.3})
  
Deep Learning

def evaluate(model, X_train, y_train, X_test, y_test, history):
    
    # Loss during Training
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.show()
    
    # loss of test data
    print("\nNeural Network Loss on Testing Data: ",model.evaluate(X_test, y_test, verbose=False))

    # R2 score on training data
    y_train_pred = model.predict(X_train, verbose=False)
    r2_train = round(r2_score(y_train_pred, y_train), 2)
    print("Neural Network R2 Score on Training Data: ", r2_train)

    # R2 score on test data
    y_val_pred = model.predict(X_val, verbose=False)
    r2_test = round(r2_score(y_val_pred, y_val), 2)
    print("Neural Network R2 Score on Testing Data: ", r2_test)
    
    return r2_train, r2_test
    
Defining Callbacks

Early Stopping We define an early stopping callback (early_stopping) to prevent overfitting during model training. This callback monitors the validation loss (val_loss) and stops training if the loss does not improve for a certain number of epochs (patience). The parameter min_delta specifies the minimum change in validation loss to be considered an improvement. By setting restore_best_weights=True, the model's weights are restored to the best performing configuration on the validation set.

Learning Rate Decay We set up a learning rate decay schedule (learning_rate_schedule) to adjust the learning rate during training. It starts with an initial learning rate of 0.01 and decreases exponentially over time. The decay_steps parameter defines how often the learning rate should decay, while decay_rate specifies the rate at which it decreases. Setting staircase=True makes the decay occur at discrete intervals.

# define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=10, restore_best_weights=True)

# learning rate decay
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=500,
    decay_rate=0.96,
    staircase=True
)

# Training model using Deep Learning Keras Library

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense

# Neural Network Architecture
nn_day = Sequential(
    [
        Dense(64, input_dim=X_train_day.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='linear'),
        Dense(1, activation='linear')
    ]
)

# print model summary
nn_day.summary()

# Compile the model
nn_day.compile(loss='mean_squared_error', optimizer='adam')

# Model Training 
history_day = nn_day.fit(X_train_day, y_train_day, validation_data=(X_val_day, y_val_day), epochs=100, verbose=None, callbacks=[early_stopping])

# Evaluate the model
nn_r2_train, nn_r2_val = evaluate(nn_day, X_train_day, y_train_day, X_val_day, y_val_day, history_day)

train_acc_day['Neural Network'] = nn_r2_train
val_acc_day['Neural Network'] = nn_r2_val

# Neural Network Architecture
nn_hour = Sequential(
    [   
        Dense(64, input_dim=X_train_hour.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='linear'),
        Dense(1, activation='linear'),
    ]
)

# print model summary
nn_hour.summary()

# Compile the model
nn_hour.compile(optimizer=Adam(learning_rate_schedule), loss='mean_squared_error')

# Model Training 
history_hour = nn_hour.fit(X_train_hour, y_train_hour, validation_data=(X_val_hour, y_val_hour), epochs=100, verbose=None, callbacks=[early_stopping], batch_size=128)

# Evaluate the model
nn_r2_train, nn_r2_val = evaluate(nn_hour, X_train_hour, y_train_hour, X_val_hour, y_val_hour, history_hour)

train_acc_hour['Neural Network'] = nn_r2_train
val_acc_hour['Neural Network'] = nn_r2_val

Model Selection

# set the format of pandas numbers
pd.options.display.float_format = '{:.2f}'.format

# dataframe of the scores obtained by models
df_scores = pd.DataFrame({"Day - Training Scores": train_acc_day,
                         "Day - Validation Scores": val_acc_day,
                         "Hour - Training Scores": train_acc_hour,
                         "Hour - Validation Scores": val_acc_hour})

# Function to color the maximum cell in each column
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lightblue' if v else '' for v in is_max]

styled_df = df_scores.style.apply(highlight_max)

styled_df

Random Forest, and XGBoost consistently perform well across both the Day and Hour datasets, indicating robustness and generalization capability.

XGBoost is the best performing model on both datasets. let's improve it by searching for the best hyperparameters.

# Concatenate training and validation data
X_train_day = np.concatenate((X_train_day, X_val_day), axis=0)
y_train_day = np.concatenate((y_train_day, y_val_day), axis=0)

# apply hyperparameter tunning
param_grid_day={'n_estimators': [50, 75, 100], 'max_depth': [4,5,6]}
model_day = tune_and_evaluate(XGBRegressor(random_state=42), param_grid=param_grid_day, X=X_train_day, y=y_train_day)

# model performance on testing data
print("Model R2 Score on Testing Data: ",round(model_day.score(X_val_day, y_val_day), 2))

# Concatenate training and validation data
X_train_hour = np.concatenate((X_train_hour, X_val_hour), axis=0)
y_train_hour = np.concatenate((y_train_hour, y_val_hour), axis=0)

# apply hyperparameter tunning
param_grid_hour={'n_estimators': [150, 200, 250], 'max_depth': [5,6,7,8,9]}
model_hour = tune_and_evaluate(XGBRegressor(random_state=42), param_grid=param_grid_hour, X=X_train_hour, y=y_train_hour)

# model performance on testing data
print("Model R2 Score on Testing Data: ",round(model_hour.score(X_val_hour, y_val_hour), 2))

import pickle 

# save day dataset model
with open("model_day.pkl", 'wb') as file:
    pickle.dump(model_day, file)
    
