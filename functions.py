#!/usr/bin/env python
# coding: utf-8

# In[1]:
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
from scipy.stats import pearsonr


def getDataFrame(url: str) -> pd.DataFrame:
    """
      This function aims to download data from the website throng API
      :param url: the url of the dataset
      :return: Returning data as a dataframe

      >>> getDataFrame('https://ghoapi.azureedge.net/api/WHOSIS_000002').shape[0]
      2328

    """
    data = requests.get(url).json()
    data = data['value']
    data = pd.DataFrame(data)
    return data


def Integrity_Check(df: pd.DataFrame):
    """
       This function aims to give an overview of the dataframe.

       :param df: the dataframe that will be checked
    """
    print('Is there any null value in the dataset')
    print(df.isna().all().all())
    print('Every year has all country data?')
    print(df.groupby(['year'])['countryCode'].nunique())
    print('which country provide more data?')
    print(df.groupby(['countryCode'])['year'].count().sort_values(ascending=False))
    print('How many cause of death in the dataset? Any herirarchy?')
    print(list[df['diseaseName'].unique()])
    print('which cause of death has more country data? ')
    print(df.groupby(['diseaseName'])['countryCode'].nunique().sort_values(ascending=False))


# In[4]:
def world_mortality_trend(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    To present a line chart shows the global death rate(the number of death per 1000 population) trend from 2000 to 2019.
    Here, we calculate the total number of death all over the world, then divided by the population in the
    whole world which is calculated by the population of each country
    :param all_data: The dataframe contains death number and population of each country of each year from 2000 to 2019.
    :return: the dataframe after groupby.

    >>> test_data = [['HTI',2015,9949318,211608],['HTI',2016,9953413,211300],['HTI',2017,99553231,300956],['LSO',2015,2018355,98354],['LSO',2016,2018355,95603],['LSO',2017,2018355,73283],['BWA',2015,1734387,15560],['BWA',2016,1775969,11247],['BWA',2017,1799472,79852]]

    >>> df = pd.DataFrame(test_data,columns=['countryCode','year','population','deathNum'])
    >>> df_after=world_mortality_trend(df)
    >>> round(df_after.loc[df_after['year'] == 2017,'death_per1000']).tolist()
    [4.0]
    """
    worldly_mortality_trend = all_data.groupby(['year', 'countryCode'], as_index=False)[['population', 'deathNum']].agg(
        {'population': np.max, 'deathNum': np.sum})
    worldly = worldly_mortality_trend.groupby(['year'], as_index=False)[['population', 'deathNum']].agg(
        {'population': np.sum, 'deathNum': np.sum})
    worldly['death_per1000'] = worldly['deathNum'] / worldly['population'] * 1000

    fig = plt.figure(figsize=(14, 7))
    plt.plot(worldly['year'],
             worldly['death_per1000'],
             linestyle='-',
             linewidth=2,
             color='steelblue',

             markeredgecolor='black',
             markerfacecolor='brown')

    plt.title('Globally mortality rate')
    plt.xlabel('Year')
    plt.ylabel('rate( per 1000 population)')
    plt.ylim((4, 10))
    plt.xticks(range(2000, 2020, 1))
    return worldly


# In[5]:
def ratetrend_incomegroup(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Present the global death rate of the 4 income groups from 2000-2019.
    death rate:the number of death per 1000 population
    :param dataset:The dataframe contains death number and population of each country of each year from 2000 to 2019.
                   Also contains the label:incomeGroup
    :return:  the dataframe after groupby/ pivot.
    >>> test_data = [['HTI',2015,9949318,211608,'L'],['HTI',2016,9953413,211300,'L'],['HTI',2017,99553231,300956,'L'],['LSO',2015,2018355,98354,'L'],['LSO',2016,2018355,95603,'L'],['LSO',2017,2018355,73283,'L'],['BWA',2015,1734387,15560,'H'],['BWA',2016,1775969,11247,'H'],['BWA',2017,1799472,79852,'H']]
    >>> df = pd.DataFrame(test_data,columns=['countryCode','year','population','deathNum','incomeGroup'])
    >>> df_after=ratetrend_incomegroup(df)
    >>> '..' in df_after['incomeGroup']
    False
    >>> round(df_after.loc[df_after['incomeGroup'] == 'L' ,'death_per1000'],2).tolist()
    [25.9, 25.64, 3.68]

    """
    causebyincome = \
        dataset.sort_values(['deathNum'], ascending=False).groupby(['incomeGroup', 'year', 'countryCode'],
                                                                   as_index=False)[
            ['population', 'deathNum']].agg({'population': np.max, 'deathNum': np.sum})
    causebyincome = causebyincome.groupby(['incomeGroup', 'year'], as_index=False)[['population', 'deathNum']].agg(
        {'population': np.sum, 'deathNum': np.sum})
    causebyincome['death_per1000'] = causebyincome['deathNum'] / causebyincome['population'] * 1000
    causebyincome = causebyincome[causebyincome['incomeGroup'] != '..']
    return causebyincome


# In[6]:
def Topdeathtype_incomegroup_year(dataset: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    To present a bar chart to show the distribution of three Mortality type in four incomegroups.
    :param dataset: The dataframe contains death number, incomegroup label, year label, and Mortality type label.
    :param year: determine the year that going to be analyzed.
    :return:the dataframe after selection and groupby...
    >>> test_data = [['HTI',2015,9949318,211608,'L','noncommunicable'],['HTI',2015,9949318,211300,'L','communicable'],['HTI',2015,9949318,300956,'L','Injuries'],['LSO',2015,2018355,98354,'L','noncommunicable'],['LSO',2016,2018355,95603,'L','communicable'],['LSO',2017,2018355,73283,'L','Injuries'],['BWA',2015,1734387,15560,'H','noncommunicable'],['BWA',2015,1775969,11247,'H','communicable'],['BWA',2015,1799472,79852,'H','Injuries']]
    >>> df = pd.DataFrame(test_data,columns=['countryCode','year','population','deathNum','incomeGroup','mortality_type'])
    >>> df_after=Topdeathtype_incomegroup_year(df,2015)
    >>> df_after.query('incomeGroup == ["L"]').values
    array([[30.24890751, 21.23763659, 25.89993894]])
    """
    causebyincome = dataset[(dataset['year'] == year)].sort_values(['deathNum'], ascending=False).groupby(
        ['incomeGroup', 'mortality_type', 'countryCode'], as_index=False)[['population', 'deathNum']].agg(
        {'population': np.max, 'deathNum': np.sum})

    causebyincome1 = causebyincome.groupby(['incomeGroup', 'mortality_type'], as_index=False)[
        ['population', 'deathNum']].agg({'population': np.sum, 'deathNum': np.sum})
    causebyincome1['death_per1000'] = causebyincome1['deathNum'] / causebyincome1['population'] * 1000
    causebyincome1 = causebyincome1[causebyincome1['incomeGroup'] != '..']
    causebyincome1 = causebyincome1.pivot(index='incomeGroup', columns='mortality_type', values='death_per1000')
    colors = ["#006D2C", "#31A354", "#74C476"]
    causebyincome1.loc[:, ['Injuries', 'communicable', 'noncommunicable']].plot.bar(stacked=True, color=colors,
                                                                                    figsize=(10, 7), ylim=([0, 10]))
    return causebyincome1


def deathtypetrend(dataset: pd.DataFrame, income_group: str = 'ALL') -> pd.DataFrame:
    """
    present a line chart to show the death rate trend of three Mortality type in the whole world or in a specifc
    income group. The trend year range is from 2000-2019, death rate= total number of death per 1000 population.
    There is one default parameter -- income_group refers to the four income groups(H,UM,LM,L), if the parameter
    not be declared, we assume the user want to analyze data in the whole range.
    P.S. For those countries not labeled as any income group, we are not going to analyze them.

    :param dataset: the compliant dataframe
    :param income_group: one of the four types of income group(H,UM,LM,L)
    :return: the dataframe after selection and groupby...
    >>> test_data = [['HTI',2015,9949318,211608,'L','noncommunicable'],['HTI',2015,9949318,211300,'L','communicable'],['HTI',2015,9949318,300956,'L','Injuries'],['LSO',2015,2018355,98354,'L','noncommunicable'],['LSO',2016,2018355,95603,'L','communicable'],['LSO',2017,2018355,73283,'L','Injuries'],['BWA',2015,1734387,15560,'H','noncommunicable'],['BWA',2015,1775969,11247,'H','communicable'],['BWA',2015,1799472,79852,'H','Injuries']]
    >>> df = pd.DataFrame(test_data,columns=['countryCode','year','population','deathNum','incomeGroup','mortality_type'])
    >>> df_after_ALL=deathtypetrend(df)
    >>> df_after_ALL.query('year == 2015').values
    array([[32.41252929, 18.98008978, 23.75715768]])

    >>> df_after_ALL=deathtypetrend(df,'L')
    >>> df_after_ALL.query('year == 2015').values
    array([[30.24890751, 21.23763659, 25.89993894]])
    """

    if income_group == 'ALL':
        income_Group = ['..', 'H', 'L', 'UM', 'LM']
    else:
        income_Group = [income_group]
    causebyincome = \
        dataset[dataset['incomeGroup'].isin(income_Group)].sort_values(['deathNum'], ascending=False).groupby(
            ['year', 'mortality_type', 'countryCode'], as_index=False)[['population', 'deathNum']].agg(
            {'population': np.max, 'deathNum': np.sum})

    causebyincome1 = causebyincome.groupby(['year', 'mortality_type'], as_index=False)[['population', 'deathNum']].agg(
        {'population': np.sum, 'deathNum': np.sum})
    causebyincome1['death_per1000'] = causebyincome1['deathNum'] / causebyincome1['population'] * 1000
    causebyincome1 = causebyincome1.pivot(index='year', columns='mortality_type', values='death_per1000')
    causebyincome1.plot(kind='line', xlabel='year', ylabel='death_per1000',
                        title='three causes of death trend from 2000-2019 in the income_group=' + income_group,
                        figsize=(10, 7), legend=True, xticks=(range(2000, 2020, 1)))
    return causebyincome1


# In[7]:


def Topcause_year(dataset: pd.DataFrame, year: int, income_group: str = 'All', category: str = 'All') -> pd.DataFrame:
    """
    To present a bar chart to show  10 leading cause of death in descending order of one specific year, ranking by the
    number of deaths.
    There are two default parameters, income_group refers to the income group, category refers to the Mortality type.
    If the parameter not be declared, we assume the user want to analyze data in the whole range.
    :param dataset: the compliant dataframe
    :param year: The year between 2000 and 2019
    :param income_group: one of the four types of income group(H,UM,LM,L)
    :param category: one of three Mortality types(communicable,noncommunicable,Injuries)
    :return: the dataframe after selection and groupby...

    >>> test_data = [['HTI',2015,9949318,211608,'L','communicable','a'],['HTI',2015,9949318,211300,'L','communicable','b'],['HTI',2015,9949318,300956,'L','Injuries','c'],['LSO',2015,2018355,98354,'L','communicable','a'],['LSO',2016,2018355,95603,'L','communicable','b'],['LSO',2017,2018355,73283,'L','Injuries','c'],['BWA',2015,1734387,15560,'H','communicable','a'],['BWA',2015,1775969,11247,'H','communicable','b'],['BWA',2015,1799472,79852,'H','Injuries','c']]
    >>> df = pd.DataFrame(test_data,columns=['countryCode','year','population','deathNum','incomeGroup','mortality_type','diseaseName'])
    >>> df_after_ALL=Topcause_year(df,2015)
    >>> df_after_ALL['diseaseName'].head(1).tolist()
    ['c']
    >>> df_after_Low=Topcause_year(df,2015,income_group='L')
    >>> df_after_Low['diseaseName'].head(1).tolist()
    ['a']
    """
    if income_group == 'All':
        incomegroup = ['H', 'L', 'UM', 'LM']
    else:
        incomegroup = [income_group]
    if category == 'All':
        mortalitycategory = ['communicable', 'noncommunicable', 'Injuries']
    else:
        mortalitycategory = [category]

    df = dataset[(dataset['year'] == year) & (dataset['incomeGroup'].isin(incomegroup)) & (
        dataset['mortality_type'].isin(mortalitycategory))].sort_values(['deathNum'], ascending=False).groupby(
        ['diseaseName'], as_index=False)[['deathNum', 'mortality_type']].agg(
        {'deathNum': np.sum, 'mortality_type': np.max}).nlargest(10, 'deathNum')

    plt.bar(df['diseaseName'], df['deathNum'])
    plt.xticks(rotation=270)
    plt.title('10 leading disease\n income group=' + income_group + ', Mortality type=' + category)
    plt.xlabel('Disease')
    plt.ylabel('Number of death')
    plt.figure(figsize=(20,20))
    plt.show()
    return df


def Topcause_trend(dataset: pd.DataFrame, year: int, income_group: str = 'All', category: str = 'All') -> pd.DataFrame:
    """
    To present a line chart to show the trend (2000-2019) of 10 leading cause of death in a specific year. The leading
    cause of death of a specific year is calculated by the function 'Topcause_year'
    There are two default parameters, income_group refers to the income group, category refers to the Mortality type.
    If the parameter not be declared, we assume the user want to analyze data in the whole range.
    :param dataset: the compliant dataframe
    :param year: The year between 2000 and 2019, aims to retrieve the 10 leading cause of deaths of this year.
    :param income_group: one of the four types of income group(H,UM,LM,L)
    :param category: one of three Mortality types(communicable,noncommunicable,Injuries)
    :return: the dataframe after selection and groupby...

    >>> test_data = [['HTI',2015,9949318,211608,'L','communicable','a'],['HTI',2016,9949318,211300,'L','communicable','a'],['HTI',2015,9949318,300956,'L','Injuries','c'],['LSO',2016,2018355,98354,'L','communicable','a'],['LSO',2016,2018355,95603,'L','communicable','b'],['LSO',2015,2018355,73283,'L','Injuries','c'],['BWA',2015,1734387,15560,'H','communicable','d'],['BWA',2016,1775969,11247,'H','communicable','b'],['BWA',2015,1799472,79852,'H','Injuries','c']]
    >>> df = pd.DataFrame(test_data,columns=['countryCode','year','population','deathNum','incomeGroup','mortality_type','diseaseName'])
    >>> df_after_ALL=Topcause_trend(df,2015)
    >>> df_after_ALL.query('year==2015')['a'].values
    array([21.26859349])
    >>> df_after_ALL.query('year==2015')['d'].values
    array([8.97146946])
    >>> df_after_Low=Topcause_trend(df,2015,income_group='L')
    >>> df_after_Low.query('year==2015')['d'].values
    Traceback (most recent call last):
    KeyError: 'd'
    >>> df_after_Low.query('year==2016')['a'].values
    array([25.87420295])

    >>> df_after_comm=Topcause_trend(df,2015,category='communicable')
    >>> df_after_comm.query('year ==2015')[['b','c']].values
    Traceback (most recent call last):
    KeyError: "None of [Index(['b', 'c'], dtype='object', name='diseaseName')] are in the [columns]"
    """
    top10cause = Topcause_year(dataset, year, income_group, category)
    disease_list = list(top10cause['diseaseName'])

    if income_group == 'All':
        incomegroup = ['H', 'L', 'UM', 'LM']
    else:
        incomegroup = [income_group]
    if category == 'All':
        mortalitycategory = ['communicable', 'noncommunicable', 'Injuries']
    else:
        mortalitycategory = [category]

    df = dataset[(dataset['incomeGroup'].isin(incomegroup)) & (dataset['mortality_type'].isin(mortalitycategory)) & (
        dataset['diseaseName'].isin(disease_list))].groupby(['diseaseName', 'year', 'countryCode'], as_index=False)[
        ['population', 'deathNum']].agg({'population': np.max, 'deathNum': np.sum})

    df1 = df.groupby(['diseaseName', 'year'], as_index=False)[['population', 'deathNum']].agg(
        {'population': np.sum, 'deathNum': np.sum})
    df1['death_per1000'] = df1['deathNum'] / df1['population'] * 1000
    df1 = df1.pivot(index='year', columns='diseaseName', values='death_per1000')
    plot = df1.plot(kind='line', xlabel='year', ylabel='the number of death per 1000 ',
                    title='Death rate trend from 2000-2019\nincome group=' + income_group + ' Mortality type=' + category,
                    figsize=(10, 7), legend=True, xticks=(range(2000, 2020, 1)))
    plot.legend(bbox_to_anchor=(1.5, 1))

    return df1


# In[10]:
def lineplot_time(HealthOutcomeData: pd.DataFrame):
    """
    :param HealthOutcomeData: data set of health outcome data: Under5_Mortality, LifeExpectancy
    :return: reformatted data set
    >>> a = [[24614321, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2015, 'SEX', 'MLE', None, None, None, None, None, None, '61.0', 61.03658, None, None, None, '2020-12-04T16:59:43.423+01:00', '2015', '2015-01-01T00:00:00+01:00', '2015-12-31T00:00:00+01:00'], [24614325, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2019, 'SEX', 'MLE', None, None, None, None, None, None, '63.3', 63.28709, None, None, None, '2020-12-04T16:59:43.533+01:00', '2019', '2019-01-01T00:00:00+01:00', '2019-12-31T00:00:00+01:00'], [24614313, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2000, 'SEX', 'MLE', None, None, None, None, None, None, '54.6', 54.57449, None, None, None, '2020-12-04T16:59:42.513+01:00', '2000', '2000-01-01T00:00:00+01:00', '2000-12-31T00:00:00+01:00'], [24614317, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2010, 'SEX', 'MLE', None, None, None, None, None, None, '59.6', 59.60036, None, None, None, '2020-12-04T16:59:43.013+01:00', '2010', '2010-01-01T00:00:00+01:00', '2010-12-31T00:00:00+01:00'], [24614329, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2000, 'SEX', 'FMLE', None, None, None, None, None, None, '55.4', 55.41726, None, None, None, '2020-12-04T16:59:43.61+01:00', '2000', '2000-01-01T00:00:00+01:00', '2000-12-31T00:00:00+01:00'], [24614333, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2010, 'SEX', 'FMLE', None, None, None, None, None, None, '60.3', 60.2972, None, None, None, '2020-12-04T16:59:43.72+01:00', '2010', '2010-01-01T00:00:00+01:00', '2010-12-31T00:00:00+01:00'], [24614337, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2015, 'SEX', 'FMLE', None, None, None, None, None, None, '62.3', 62.34584, None, None, None, '2020-12-04T16:59:43.847+01:00', '2015', '2015-01-01T00:00:00+01:00', '2015-12-31T00:00:00+01:00'], [24614341, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2019, 'SEX', 'FMLE', None, None, None, None, None, None, '63.2', 63.15551, None, None, None, '2020-12-04T16:59:43.927+01:00', '2019', '2019-01-01T00:00:00+01:00', '2019-12-31T00:00:00+01:00'], [24614345, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2000, 'SEX', 'BTSX', None, None, None, None, None, None, '55.0', 54.98949, None, None, None, '2020-12-04T16:59:44.053+01:00', '2000', '2000-01-01T00:00:00+01:00', '2000-12-31T00:00:00+01:00'], [24614349, 'WHOSIS_000001', 'COUNTRY', 'AFG', 'YEAR', 2010, 'SEX', 'BTSX', None, None, None, None, None, None, '59.9', 59.94055, None, None, None, '2020-12-04T16:59:44.18+01:00', '2010', '2010-01-01T00:00:00+01:00', '2010-12-31T00:00:00+01:00']]
    >>> cols = ['Id', 'IndicatorCode', 'SpatialDimType', 'SpatialDim', 'TimeDimType', 'TimeDim', 'Dim1Type', 'Dim1', 'Dim2Type', 'Dim2', 'Dim3Type', 'Dim3', 'DataSourceDimType', 'DataSourceDim', 'Value', 'NumericValue', 'Low', 'High', 'Comments', 'Date', 'TimeDimensionValue', 'TimeDimensionBegin', 'TimeDimensionEnd']
    >>> df = pd.DataFrame(a, columns = cols)
    >>> lineplot_time("Under5_Mortality_1")
    Traceback (most recent call last):
    AttributeError: 'str' object has no attribute 'columns'
    
    >>> lineplot_time(df[['TimeDim']])
    Traceback (most recent call last):
    KeyError

    """
    if ('TimeDim' not in HealthOutcomeData.columns) or ('NumericValue' not in HealthOutcomeData.columns) or ('SpatialDim' not in HealthOutcomeData.columns):
        raise KeyError
    b = HealthOutcomeData[(HealthOutcomeData['Dim1'] == 'BTSX') & (HealthOutcomeData['SpatialDimType'] == 'REGION')][['TimeDim','NumericValue','SpatialDim']]
    a = b[b['SpatialDim'] != 'GLOBAL'] # no global value
    a = a.astype({"TimeDim": str})
    a = a.pivot(index='TimeDim', columns='SpatialDim',values = 'NumericValue')
    a = a.rename_axis(None, axis=1)
    plt.figure(figsize=(16,8))
    p = sns.lineplot(data=a)
    p.set(xlabel = "Year", ylabel = 'Value')
    sns.set(rc={"figure.figsize":(3, 4)})
    plt.show()



# In[11]:
def formatWithSex(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    only applies to life expectancy and Under5_Mortality
    :param name: reformat with sex, pivot that column
    :param df:
    :return:
    >>> Under5_Mortality_1 = [[27818216, 'MDG_0000000007', 'REGION', 'AFR', 'YEAR', 1990, 'SEX','BTSX', None, None, None, None, None, None,'176.2 [172.72-180.17]', 176.20446, 172.71708, 180.16938, None, '2022-01-18T13:13:12.597+01:00', '1990','1990-01-01T00:00:00+01:00', '1990-12-31T00:00:00+01:00'],[27818217, 'MDG_0000000007', 'REGION', 'AFR', 'YEAR', 1990, 'SEX',  'FMLE', None, None, None, None, None, None, '167.03 [163.62-170.86]', 167.02559, 163.62235, 170.85607, None, '2022-01-18T13:13:12.613+01:00', '1990','1990-01-01T00:00:00+01:00', '1990-12-31T00:00:00+01:00'],[27818218, 'MDG_0000000007', 'REGION', 'AFR', 'YEAR', 1990, 'SEX', 'MLE', None, None, None, None, None, None,'184.95 [181.2-189.23]', 184.95174, 181.20198, 189.22685, None,'2022-01-18T13:13:12.63+01:00', '1990', '1990-01-01T00:00:00+01:00', '1990-12-31T00:00:00+01:00'],[27818219, 'MDG_0000000007', 'REGION', 'AFR', 'YEAR', 1991, 'SEX', 'BTSX', None, None, None, None, None, None, '174.86 [171.44-178.75]', 174.85804, 171.4446, 178.752, None,'2022-01-18T13:13:12.643+01:00', '1991','1991-01-01T00:00:00+01:00', '1991-12-31T00:00:00+01:00'],[27818220, 'MDG_0000000007', 'REGION', 'AFR', 'YEAR', 1991, 'SEX', 'FMLE', None, None, None, None, None, None, '165.7 [162.3-169.42]', 165.70279, 162.30395, 169.42294, None, '2022-01-18T13:13:12.66+01:00', '1991','1991-01-01T00:00:00+01:00', '1991-12-31T00:00:00+01:00']]
    >>> type(formatWithSex('Under5_Mortality', Under5_Mortality_1)['year'][0])
    Traceback (most recent call last):
    TypeError: list indices must be integers or slices, not str
    """
    df = df[df['SpatialDimType'] == 'COUNTRY']
    df = df[['SpatialDim','TimeDim','Dim1','NumericValue']]
    df = df.rename(columns={"SpatialDim":"countrycode", "TimeDim":"year", "Dim1":"Sex","NumericValue": 'df'})
    df = df.pivot(index=['countrycode','year'],columns = ['Sex'],values= 'df')
    df = df.reset_index()
    df.rename(columns={'BTSX':f'{name}_BTSX','FMLE':f'{name}_FMLE','MLE':f'{name}_MLE'},inplace=True)
    df = df[['countrycode','year',f'{name}_BTSX',f'{name}_FMLE',f'{name}_MLE']]
    df = df.astype({"year": str})
    df = df.rename_axis(None, axis=1) # remove "Sex" as the index name
    return df


# In[12]:


def formatWithoutSex(df: pd.DataFrame) -> pd.DataFrame:
    """
    only applies to maternal mortality rate
    :param df: the target data set
    :return: the reformatted data set
    >>> MaternalMortalityRatio_1 = pd.DataFrame()
    >>> formatWithoutSex(MaternalMortalityRatio_1)
    Traceback (most recent call last):
    KeyError: 'SpatialDimType'
    """
    df = df[df['SpatialDimType'] == 'COUNTRY']
    df = df[['SpatialDim','TimeDim','NumericValue']]
    df = df.rename(columns={"SpatialDim":"countrycode", "TimeDim":"year", "NumericValue": 'MaternalMortalityRatio'})
    df = df.astype({"year": str})
    df = df[['countrycode','year','MaternalMortalityRatio']]
    return df


def effectOfexp2015before(data: pd.DataFrame,col: str,region: str) -> tuple:
    """
    correlation before 2015
    :param data: data set that contains MaternalMortalityRatio, or Under5_Mortality, or LifeExpectancy information
    :param col: the corresponding MaternalMortalityRatio, or Under5_Mortality, or LifeExpectancy column, to calculate the correlation between che_gdp and the input corresponded column
    :param region: one of the region
    :return: coefficient, pvalue
    >>> Under5_clean = [['Afghanistan', 'South Asia', 'AFG', 2002, 'L', 9.44338989,121.06367],['Afghanistan', 'South Asia', 'AFG', 2003, 'L', 8.941258430000001,116.91188],['Afghanistan', 'South Asia', 'AFG', 2004, 'L', 9.80847359,112.63116],['Afghanistan', 'South Asia', 'AFG', 2005, 'L', 9.94828987, 108.38065],['Afghanistan', 'South Asia', 'AFG', 2006, 'L', 10.62276554,104.11641], ['Afghanistan', 'South Asia', 'AFG', 2007, 'L', 9.90467453, 99.89215],['Afghanistan', 'South Asia', 'AFG', 2008, 'L',  10.256495480000002, 95.71579],['Afghanistan', 'South Asia', 'AFG', 2009, 'L', 9.81848717, 91.68161],['Afghanistan', 'South Asia', 'AFG', 2010, 'L', 8.56967163,  87.75619],['Afghanistan', 'South Asia', 'AFG', 2011, 'L', 8.561906809999998, 83.97012]]
    >>> cols = ['Country','Region','CountryCode','Year','incomeGroup','che_gdp', 'Under5_Mortality_BTSX']
    >>> Under5_clean = pd.DataFrame(Under5_clean, columns = cols)
    >>> effectOfexp2015before(Under5_clean,'Under5_Mortality_BTSX','Europe & Central')
    (None, None)
    """
    if region not in ['East Asia & Pacific', 'Europe & Central Asia', 'Latin America & Caribbean',
                      'Middle East & North Africa', 'North America', 'South Asia', 'Sub-Saharan Africa']:
        return None, None
    data = data.copy().astype({"Year":int})
    cleaned = data.loc[
        (data["Year"] == 2000) | (data["Year"] == 2001) | (data["Year"] == 2002) | (data["Year"] == 2003) | (
                data["Year"] == 2004) | (data["Year"] == 2005) | (data["Year"] == 2006) | (data["Year"] == 2007) | (
                data["Year"] == 2008) | (data["Year"] == 2009) | (data["Year"] == 2010) | (data["Year"] == 2011) | (
                data["Year"] == 2012) | (data["Year"] == 2013) | (data["Year"] == 2014)]
    coefficient, pvalue = pearsonr(cleaned[cleaned['Region'] == region]['che_gdp'],cleaned[cleaned['Region'] == region][col])
    return coefficient, pvalue


def effectOfexp2015after(data: pd.DataFrame,col: str,region: str) -> tuple:
    """
    correlation after 2015
    :param data: data set that contains MaternalMortalityRatio, or Under5_Mortality, or LifeExpectancy information
    :param col: the corresponding MaternalMortalityRatio, or Under5_Mortality, or LifeExpectancy column, to calculate the correlation between che_gdp and the input corresponded column
    :param region: one of the region
    :return: coefficient, pvalue
    """
    data = data.copy().astype({"Year":int})
    if region in ['East Asia & Pacific', 'Europe & Central Asia', 'Latin America & Caribbean',
                  'Middle East & North Africa', 'North America', 'South Asia', 'Sub-Saharan Africa']:
        cleaned = data.loc[
            (data["Year"] == 2015) | (data["Year"] == 2016) | (data["Year"] == 2017) | (data["Year"] == 2018) | (
                    data["Year"] == 2019)]
        coefficient, pvalue = pearsonr(cleaned[cleaned['Region'] == region]['che_gdp'],
                                       cleaned[cleaned['Region'] == region][col])
        return coefficient, pvalue
    else:
        return
