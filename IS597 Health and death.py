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
    >>> df_after_ALL
    """

    if income_group == 'ALL':
        incomegroup = ['..', 'H', 'L', 'UM', 'LM']
    else:
        incomegroup = [income_group]
    causebyincome = \
        dataset[dataset['incomeGroup'].isin(incomegroup)].sort_values(['deathNum'], ascending=False).groupby(
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
def lineplot_time(HealthOutcomeData):
    """
    :param HealthOutcomeData: data set of health outcome data: Under5_Mortality, LifeExpectancy
    :return:
    """
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
def formatWithSex(name, df):
    """
    only applies to life expectancy and Under5_Mortality
    :param name: reformat with sex, pivot that column
    :param df:
    :return:
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


def formatWithoutSex(df):
    """
    only applies to maternal mortality rate
    :param df: the target data set
    :return: the reformatted data set
    """
    df = df[df['SpatialDimType'] == 'COUNTRY']
    df = df[['SpatialDim', 'TimeDim', 'NumericValue']]
    df = df.rename(columns={"SpatialDim": "countrycode", "TimeDim": "year", "NumericValue": 'MaternalMortalityRatio'})
    df = df.astype({"year": str})
    df = df[['countrycode', 'year', 'MaternalMortalityRatio']]
    return df


def effectOfexp2015before(data, col, region):
    """
    correlation before 2015
    :param data: data set that contains MaternalMortalityRatio, or Under5_Mortality, or LifeExpectancy information
    :param col: the corresponding MaternalMortalityRatio, or Under5_Mortality, or LifeExpectancy column, to calculate the correlation between che_gdp and the input corresponded column
    :param region: one of the region
    :return: coefficient, pvalue
    """

    if region not in ['East Asia & Pacific', 'Europe & Central Asia', 'Latin America & Caribbean',
                      'Middle East & North Africa', 'North America', 'South Asia', 'Sub-Saharan Africa']:
        return
    cleaned = data.loc[
        (data["Year"] == 2000) | (data["Year"] == 2001) | (data["Year"] == 2002) | (data["Year"] == 2003) | (
                data["Year"] == 2004) | (data["Year"] == 2005) | (data["Year"] == 2006) | (data["Year"] == 2007) | (
                data["Year"] == 2008) | (data["Year"] == 2009) | (data["Year"] == 2010) | (data["Year"] == 2011) | (
                data["Year"] == 2012) | (data["Year"] == 2013) | (data["Year"] == 2014)]
    coefficient, pvalue = pearsonr(cleaned[cleaned['Region'] == region]['che_gdp'],cleaned[cleaned['Region'] == region][col])
    return coefficient, pvalue


def effectOfexp2015after(data, col, region):
    """
    correlation after 2015
    :param data: data set that contains MaternalMortalityRatio, or Under5_Mortality, or LifeExpectancy information
    :param col: the corresponding MaternalMortalityRatio, or Under5_Mortality, or LifeExpectancy column, to calculate the correlation between che_gdp and the input corresponded column
    :param region: one of the region
    :return: coefficient, pvalue
    """
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




if __name__ == '__main__':

    LifeExpectancy_1 = getDataFrame('https://ghoapi.azureedge.net/api/WHOSIS_000001')
    Under5_Mortality_1 = getDataFrame('https://ghoapi.azureedge.net/api/MDG_0000000007')
    MaternalMortalityRatio_1 = getDataFrame('https://ghoapi.azureedge.net/api/MDG_0000000026')
    global_data = getDataFrame(
        'https://frontdoor-l4uikgap6gz3m.azurefd.net/DEX_CMS/GHE_FULL?&$orderby=VAL_DEATHS_RATE100K_NUMERIC%20desc&$select=DIM_COUNTRY_CODE,DIM_GHECAUSE_CODE,DIM_GHECAUSE_TITLE,DIM_YEAR_CODE,DIM_SEX_CODE,DIM_AGEGROUP_CODE,VAL_DALY_COUNT_NUMERIC,VAL_DEATHS_COUNT_NUMERIC,ATTR_POPULATION_NUMERIC,VAL_DALY_RATE100K_NUMERIC,VAL_DEATHS_RATE100K_NUMERIC&$filter=FLAG_RANKABLE%20eq%201%20and%20DIM_SEX_CODE%20eq%20%27BTSX%27%20and%20DIM_AGEGROUP_CODE%20eq%20%27ALLAges%27')

    incomegroup = pd.read_excel("datasources/income group.xlsx", sheet_name='Sheet1')
    region = pd.read_csv("datasources/country_region.csv")[['Country', 'Region']]
    healthExp = pd.read_excel("datasources/healthExp_data.xlsx", sheet_name='cleaned')
    healthExp = healthExp[['country', 'year', 'che_gdp']]

    incomegroup = pd.melt(incomegroup, id_vars=['CountryCode', 'CountryName'],
                          value_vars=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                      2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], var_name='Year',
                          value_name='incomeGroup')
    # left join with region table.

    # merge datasets based on two columns:
    countryInfo = incomegroup.merge(region, left_on='CountryName', right_on='Country')
    countryInfo = countryInfo[['Country', 'CountryCode', 'Region', 'Year', 'incomeGroup']]
    countryInfoAll = countryInfo.merge(healthExp, left_on=['Country', 'Year'], right_on=['country', 'year'])
    countryInfoAll = countryInfoAll[['Country', 'Region', 'CountryCode', 'Year', 'incomeGroup', 'che_gdp']]
    countryInfoAll = countryInfoAll.astype({"Year":int})
    # reformatting data set for further analysis
    Under5_Mortality = formatWithSex('Under5_Mortality', Under5_Mortality_1)
    LifeExpectancy = formatWithSex('LifeExpectancy', LifeExpectancy_1)
    MaternalMortalityRatio = formatWithoutSex(MaternalMortalityRatio_1)
    InfoAll = countryInfoAll.merge(LifeExpectancy, left_on=['CountryCode', 'Year'], right_on=['countrycode', 'year'],how='left')
    InfoAll = InfoAll.merge(Under5_Mortality, left_on=['CountryCode', 'Year'], right_on=['countrycode', 'year'], how='left')
    InfoAll = InfoAll.merge(MaternalMortalityRatio, left_on=['CountryCode', 'Year'], right_on=['countrycode', 'year'],how='left')
    InfoAll = InfoAll[['Country', 'Region', 'CountryCode', 'Year', 'incomeGroup', 'che_gdp', 'LifeExpectancy_BTSX','LifeExpectancy_FMLE', 'LifeExpectancy_MLE', 'Under5_Mortality_BTSX', 'Under5_Mortality_FMLE','Under5_Mortality_MLE', 'MaternalMortalityRatio']]


    global_data_1 = global_data.rename(columns={"DIM_COUNTRY_CODE": "countryCode",
                                                "DIM_YEAR_CODE": "year",
                                                "ATTR_POPULATION_NUMERIC": "population",
                                                "DIM_GHECAUSE_CODE": "diseaseCode",
                                                "DIM_GHECAUSE_TITLE": "diseaseName",
                                                "VAL_DEATHS_COUNT_NUMERIC": "deathNum"
                                                })

    global_data_1['diseaseCode'] = global_data_1['diseaseCode'].astype('int')
    global_data_1['year'] = global_data_1['year'].astype('int')

    global_data_1 = global_data_1.merge(countryInfo, how='left', left_on=['countryCode', 'year'],
                                        right_on=['CountryCode', 'Year'])  # left join

    global_data_1['mortality_type'] = None
    global_data_1.loc[global_data_1['diseaseCode'] < 600, 'mortality_type'] = 'communicable'
    global_data_1.loc[(global_data_1['diseaseCode'] > 600) & (
            global_data_1['diseaseCode'] < 1510), 'mortality_type'] = 'noncommunicable'
    global_data_1.loc[global_data_1['diseaseCode'] > 1510, 'mortality_type'] = 'Injuries'

    # In[31]:

    global_data_1 = global_data_1[
        ['countryCode', 'Country', 'incomeGroup', 'year', 'population', 'mortality_type', 'diseaseCode', 'diseaseName',
         'deathNum']]

    # In[32]:

    Integrity_Check(global_data_1)

    # In[33]:

    world_mortality_trend(global_data_1)

    # In[34]:

    d = ratetrend_incomegroup(global_data_1)
    plotdata = d.pivot(index='year', columns='incomeGroup', values='death_per1000')
    plotdata.plot(kind='line', xlabel='year', ylabel='death_per1000 ',
                  title='death rate trend from 2000-2019 by income_group',
                  figsize=(12, 8), legend=True, xticks=(range(2000, 2020, 1)))

    # In[35]:

    deathtypetrend(global_data_1)

    # In[36]:

    print('2004')
    Topdeathtype_incomegroup_year(global_data_1, 2004)

    '''there death types, it matched the paper1:
    1. When in 2004, Communicable diseases remain an important cause of death in lowincome countries.
    2. Confirm the growing importance of noncommunicable diseases in most low- and middle-income countries.'''

    # In[37]:

    deathtypetrend(global_data_1, income_group='L')
    deathtypetrend(global_data_1, income_group='LM')
    deathtypetrend(global_data_1, income_group='UM')
    deathtypetrend(global_data_1, income_group='H')

    # In[38]:

    print('\nGlobally leading specific disease\n\n', Topcause_year(global_data_1, 2004))

    print('\n\nLow-income leading specific disease\n\n ', Topcause_year(global_data_1, 2004, income_group='L'))
    print('\n\nLow-middle income leading specific disease\n\n', Topcause_year(global_data_1, 2004, income_group='LM'))
    print('\n\nUpper-middle income leading specific disease\n\n', Topcause_year(global_data_1, 2004, income_group='UM'))
    print('\n\nHigh income leading specific disease\n\n', Topcause_year(global_data_1, 2004, income_group='H'))

    '''As may be expected from the very different distributions of deaths by
    age and sex, there are major differences in the ranking of causes
    between high- and low-income countries (Table 4). In low-income
    countries, the dominant causes are infectious and parasitic diseases
    (including malaria), and neonatal causes. In the high-income countries,
    9 of the 10 leading causes of death are non-communicable conditions,
    including the four types of cancer. In the middle-income countries, the
    10 leading causes of death are again dominated by non-communicable
    conditions; they also include road traffic accidents as the sixth most
    common cause.'''

    # In[39]:

    Topcause_trend(global_data_1, 2004, income_group='L', category='communicable')

    # In[40]:

    Topcause_trend(global_data_1, 2005, income_group='UM', category='noncommunicable')

    # In[41]:

    print(Topcause_year(global_data_1, 2004))
    print(Topcause_year(global_data_1, 2004, income_group='L', category='noncommunicable'))

    # Hypothesis 2
    # finding: the life expectancy is increasing over years
    lineplot_time(LifeExpectancy_1)

    # finding: the Under5_Mortality is decreasing over years
    lineplot_time(Under5_Mortality_1)

    # finding: the average of Maternal Mortality Ratio over the world was decreasing over time
    plt.figure(figsize=(16, 8))
    MaternalMortalityRatio = MaternalMortalityRatio.sort_values('year')
    sns.lineplot(data=MaternalMortalityRatio, x="year", sort=False, y="MaternalMortalityRatio")
    sns.set(rc={"figure.figsize": (3, 4)})
    plt.show()


    # all data includes health expendure, three factors, year, incomegroup, and region for each country
    # anova table shows that all three factors were related to health expenditure, of which the region has the most significant correlation
    # Ordinary Least Squares (OLS) model
    model = ols('che_gdp ~ Year+incomeGroup+Region', data=InfoAll).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table

    # correlation between independent Year and dependent var = che_gdp in Year 2019
    yearexp = InfoAll[['che_gdp', 'Year']].dropna()
    pearsonr(yearexp['che_gdp'], yearexp['Year'])

    # health expenditure per region per year
    InfoAll2015_ = InfoAll.loc[(InfoAll["Year"] == 2015) | (InfoAll["Year"] == 2016) | (InfoAll["Year"] == 2017) | (
                InfoAll["Year"] == 2018) | (InfoAll["Year"] == 2019)]
    d = InfoAll2015_[['Region', 'Year', 'che_gdp']]
    d = d.dropna()
    d = d.pivot_table(index='Region', columns='Year', values='che_gdp', aggfunc='mean')
    print(d)

    # generate a boxplot to see the health expenditure distribution by income groups. Using boxplot, we can easily detect the differences between income groups
    # CAN is only country in North America
    plt.figure(figsize=(16, 8))
    ax = sns.boxplot(x='Region', y='che_gdp', data=InfoAll[InfoAll['Year'] == 2019], color='#99c2a2')
    ax = sns.swarmplot(x="Region", y="che_gdp", data=InfoAll[InfoAll['Year'] == 2019], color=".2")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    sns.set(rc={"figure.figsize": (3, 4)})
    plt.show()
    # highest value: Tuvalu in East Asia

    # life Expectancy recorded data every five year. 2010, 2015, 2000, 2019.
    # drop nan before calculating correlation

    lifeExp_clean = InfoAll[['Country', 'Region', 'CountryCode', 'Year', 'incomeGroup', 'che_gdp', 'LifeExpectancy_BTSX']].dropna()
    Under5_clean = InfoAll[['Country', 'Region', 'CountryCode', 'Year', 'incomeGroup', 'che_gdp', 'Under5_Mortality_BTSX']].dropna()
    MaternalMortality_clean = InfoAll[['Country', 'Region', 'CountryCode', 'Year', 'incomeGroup', 'che_gdp', 'MaternalMortalityRatio']].dropna()

    # aligned with previous finding that in Sub-Saharan Africa Under5_Mortality in both sex decreased with health expenditure increased
    print(effectOfexp2015before(Under5_clean, 'Under5_Mortality_BTSX', 'Sub-Saharan Africa'))
    print(effectOfexp2015before(Under5_clean, 'Under5_Mortality_BTSX', 'Latin America & Caribbean'))  #
    print(effectOfexp2015before(Under5_clean, 'Under5_Mortality_BTSX', 'Europe & Central Asia'))  #
    print(effectOfexp2015before(Under5_clean, 'Under5_Mortality_BTSX', 'Middle East & North Africa'))  #
    print(effectOfexp2015before(Under5_clean, 'Under5_Mortality_BTSX', 'North America'))  #
    print(effectOfexp2015before(Under5_clean, 'Under5_Mortality_BTSX', 'South Asia'))
    print(effectOfexp2015before(Under5_clean, 'Under5_Mortality_BTSX', 'East Asia & Pacific'))

    # except for Sub-Saharan Africa and South Asia, the life expectancy increased with every increase in health expenditure
    # after 2015, in Middle East & North Africa, with every one percent increase in health expenditure over total GDP, results in 0.42 percent decrease in the Under5_Mortality rate despite of sex
    print(effectOfexp2015after(Under5_clean, 'Under5_Mortality_BTSX', 'Sub-Saharan Africa'))
    print(effectOfexp2015after(Under5_clean, 'Under5_Mortality_BTSX', 'Latin America & Caribbean'))
    print(effectOfexp2015after(Under5_clean, 'Under5_Mortality_BTSX', 'Europe & Central Asia'))
    print(effectOfexp2015after(Under5_clean, 'Under5_Mortality_BTSX', 'Middle East & North Africa'))
    print(effectOfexp2015after(Under5_clean, 'Under5_Mortality_BTSX', 'North America'))
    print(effectOfexp2015after(Under5_clean, 'Under5_Mortality_BTSX', 'South Asia'))
    print(effectOfexp2015after(Under5_clean, 'Under5_Mortality_BTSX', 'East Asia & Pacific'))

    # only one country in north america, the relationship is weird
    # conclusion from regional paper: year: before 2015 in Sub-Saharan Africa, 1 percent increase in health expenditure per capita improve life expectancy by 0.06 percent.
    # our finding: after 2015, except for Sub-Saharan Africa and South Asia, the life expectancy increased with every increase in health expenditure
    # in Latin America & Caribbean, with every one percent increase in health expenditure over total GDP, results in 0.42 years increase in life expectancy
    print(effectOfexp2015after(lifeExp_clean, 'LifeExpectancy_BTSX', 'Sub-Saharan Africa'))
    print(effectOfexp2015after(lifeExp_clean, 'LifeExpectancy_BTSX', 'Latin America & Caribbean'))
    print(effectOfexp2015after(lifeExp_clean, 'LifeExpectancy_BTSX', 'Europe & Central Asia'))
    print(effectOfexp2015after(lifeExp_clean, 'LifeExpectancy_BTSX', 'Middle East & North Africa'))
    print(effectOfexp2015after(lifeExp_clean, 'LifeExpectancy_BTSX', 'North America'))
    print(effectOfexp2015after(lifeExp_clean, 'LifeExpectancy_BTSX', 'South Asia'))
    print(effectOfexp2015after(lifeExp_clean, 'LifeExpectancy_BTSX', 'East Asia & Pacific'))

    # maternal mortality rate for each region and its correlation with the increase/decrease in health expenditure
    # except for Sub-Saharan Africa and South Asia, the mortality rate decreased with every increase in health expenditure
    # in Latin America & Caribbean, with every one percent increase in health expenditure over total GDP, results in 0.35 percent decrease in the mortality rate
    print(effectOfexp2015after(MaternalMortality_clean, 'MaternalMortalityRatio', 'Sub-Saharan Africa'))
    print(effectOfexp2015after(MaternalMortality_clean, 'MaternalMortalityRatio', 'Latin America & Caribbean'))
    print(effectOfexp2015after(MaternalMortality_clean, 'MaternalMortalityRatio', 'Europe & Central Asia'))
    print(effectOfexp2015after(MaternalMortality_clean, 'MaternalMortalityRatio', 'Middle East & North Africa'))
    print(effectOfexp2015after(MaternalMortality_clean, 'MaternalMortalityRatio', 'North America'))
    print(effectOfexp2015after(MaternalMortality_clean, 'MaternalMortalityRatio', 'South Asia'))
    print(effectOfexp2015after(MaternalMortality_clean, 'MaternalMortalityRatio', 'East Asia & Pacific'))
