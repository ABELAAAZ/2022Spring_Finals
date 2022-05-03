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


# In[2]:


def getDataFrame(url):
    data = requests.get(url).json()
    data = data['value']
    data = pd.DataFrame(data)
    return data


# In[3]:


def Integrity_Check(df):
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



def world_mortality_trend(all_data):
    worldly_mortality_trend=all_data.groupby(['year','countryCode'],as_index=False)[['population','deathNum']].agg({'population':np.max,'deathNum': np.sum})
    worldly=worldly_mortality_trend.groupby(['year'],as_index=False)[['population','deathNum']].agg({'population':np.sum, 'deathNum': np.sum})
    worldly['death_per1000']=worldly['deathNum']/worldly['population']*1000

    fig = plt.figure(figsize = (14,7))
    plt.plot(worldly['year'],
         worldly['death_per1000'],
         linestyle = '-',
         linewidth = 2,
         color = 'steelblue',

         markeredgecolor='black',
         markerfacecolor='brown')

    plt.title('Globally mortality rate')
    plt.xlabel('Year')
    plt.ylabel('rate( per 1000 population)')
    plt.ylim((4, 10))
    plt.xticks(range(2000,2020,1))
    return worldly


# In[5]:



def ratetrend_incomegroup(dataset):#done
    causebyincome=dataset.sort_values(['deathNum'],ascending=False).groupby(['incomeGroup','year','countryCode'],as_index=False)[['population','deathNum']].agg({'population':np.max,'deathNum': np.sum})
    causebyincome=causebyincome.groupby(['incomeGroup','year'],as_index=False)[['population','deathNum']].agg({'population':np.sum, 'deathNum': np.sum})
    causebyincome['death_per1000']=causebyincome['deathNum']/causebyincome['population']*1000
    causebyincome=causebyincome[causebyincome['incomeGroup']!='..']
    return causebyincome


# In[6]:



def Topdeathtype_incomegroup_year(dataset,  year):   #done
    causebyincome=dataset[(dataset['year']==year)].sort_values(['deathNum'],ascending=False).groupby(['incomeGroup','mortality_type','countryCode'],as_index=False)[['population','deathNum']].agg({'population':np.max,'deathNum': np.sum})

    causebyincome1=causebyincome.groupby(['incomeGroup','mortality_type'],as_index=False)[['population','deathNum']].agg({'population':np.sum, 'deathNum': np.sum})
    causebyincome1['death_per1000']=causebyincome1['deathNum']/causebyincome1['population']*1000
    causebyincome1=causebyincome1[causebyincome1['incomeGroup']!='..']
    causebyincome1 = causebyincome1.pivot(index='incomeGroup', columns='mortality_type', values='death_per1000')
    colors = ["#006D2C", "#31A354","#74C476"]
    causebyincome1.loc[:,['Injuries','communicable', 'noncommunicable']].plot.bar(stacked=True, color=colors, figsize=(10,7),ylim=([0,10]))
    return causebyincome1


# In[7]:


def Topcause_trend(dataset,year,income_group='All',category='All'):
    top10cause=Topcause_year(dataset,year,income_group,category)
    disease_list=list(top10cause['diseaseName'])

    if income_group=='All':
        incomegroup=['H','L','UM','LM']
    else:
        incomegroup=[income_group]
    if category=='All':
        mortalitycategory=['communicable','noncommunicable','Injuries']
    else:
        mortalitycategory=[category]

    df=dataset[(dataset['incomeGroup'].isin(incomegroup))&(dataset['mortality_type'].isin(mortalitycategory))&(dataset['diseaseName'].isin(disease_list))].groupby(['diseaseName','year','countryCode'],as_index=False)[['population','deathNum']].agg({'population':np.max,'deathNum': np.sum})

    df1=df.groupby(['diseaseName','year'],as_index=False)[['population','deathNum']].agg({'population':np.sum,'deathNum': np.sum})
    df1['death_per1000']=df1['deathNum']/df1['population']*1000
    df1 = df1.pivot(index='year', columns='diseaseName', values='death_per1000')
    plot=df1.plot(kind='line', xlabel='year',ylabel='the number of death per 1000 ',
                 title='Death rate trend from 2000-2019\nincome group='+income_group+' Mortality type='+category,
                 figsize=(10, 7),legend=True,xticks=(range(2000,2020,1)))
    plot.legend(bbox_to_anchor=(1.5, 1))

    return df1


# In[8]:


def Topcause_year(dataset, year,income_group='All',category='All'):#done
    if income_group=='All':
        incomegroup=['H','L','UM','LM']
    else:
        incomegroup=[income_group]
    if category=='All':
        mortalitycategory=['communicable','noncommunicable','Injuries']
    else:
        mortalitycategory=[category]

    df=dataset[(dataset['year']==year)&(dataset['incomeGroup'].isin(incomegroup))&(dataset['mortality_type'].isin(mortalitycategory))].sort_values(['deathNum'],ascending=False).groupby(['diseaseName'],as_index=False)[['deathNum','mortality_type']].agg({'deathNum': np.sum,'mortality_type':np.max}).nlargest(10,'deathNum')

    plt.bar(df['diseaseName'],df['deathNum'])
    plt.xticks(rotation=270)
    plt.title('10 leading disease\n income group='+income_group+', Mortality type='+category)
    plt.xlabel('Disease')
    plt.ylabel('Number of death')
    plt.show()
    return df


# In[9]:


def deathtypetrend(dataset, income_group='ALL'):
    if income_group=='ALL':
        incomegroup=['..','H','L','UM','LM']
    else:
        incomegroup=[income_group]
    causebyincome=dataset[dataset['incomeGroup'].isin(incomegroup)].sort_values(['deathNum'],ascending=False).groupby(['year','mortality_type','countryCode'],as_index=False)[['population','deathNum']].agg({'population':np.max,'deathNum': np.sum})

    causebyincome1=causebyincome.groupby(['year','mortality_type'],as_index=False)[['population','deathNum']].agg({'population':np.sum, 'deathNum': np.sum})
    causebyincome1['death_per1000']=causebyincome1['deathNum']/causebyincome1['population']*1000
    causebyincome1 = causebyincome1.pivot(index='year', columns='mortality_type', values='death_per1000')
    causebyincome1.plot(kind='line', xlabel='year',ylabel='death_per1000',
                 title='three causes of death trend from 2000-2019 in the income_group='+income_group,
                 figsize=(10, 7),legend=True,xticks=(range(2000,2020,1)))
    return causebyincome1


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
    p = sns.lineplot(data=a)
    p.set( xlabel = "Year", ylabel = 'Value')


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
    df = df.rename_axis(None, axis=1) # remove "Sex" as the index name
    return df


# In[12]:


def formatWithoutSex(df):
    df = df[df['SpatialDimType'] == 'COUNTRY']
    df = df[['SpatialDim','TimeDim','NumericValue']]
    df = df.rename(columns={"SpatialDim":"countrycode", "TimeDim":"year", "NumericValue": 'MaternalMortalityRatio'})
    df = df.astype({"year": str})
    df = df[['countrycode','year','MaternalMortalityRatio']]
    return df


# In[13]:


def effectOfexp2015before(data,col,region):
    if region not in ['East Asia & Pacific','Europe & Central Asia','Latin America & Caribbean','Middle East & North Africa','North America','South Asia','Sub-Saharan Africa']:
        return
    cleaned = data.loc[(data["Year"] == 2000) | (data["Year"] == 2001) | (data["Year"] == 2002) | (data["Year"] == 2003) | (data["Year"] == 2004) | (data["Year"] == 2005) | (data["Year"] == 2006) | (data["Year"] == 2007) | (data["Year"] == 2008) | (data["Year"] == 2009) | (data["Year"] == 2010) | (data["Year"] == 2011) | (data["Year"] == 2012) | (data["Year"] == 2013) | (data["Year"] == 2014)]
    coefficient, pvalue = pearsonr(cleaned[cleaned['Region'] == region]['che_gdp'], cleaned[cleaned['Region'] == region][col])
    return coefficient, pvalue


# In[14]:


# life expectancy: Europe & Central Asia positively correlated; Sub-Saharan Africa negatively correlated: differ in regions:
# year: before 2015 in Sub-Saharan Africa, 1 percent increase in health expenditure per capita improve life expectancy by 0.06 percent.
# in 2019: sub-Saharan Africa, 1 percent increase in health expenditure per capita improve life expectancy by 0.06 percent.
def effectOfexp2015after(data,col,region):
    if region in ['East Asia & Pacific','Europe & Central Asia','Latin America & Caribbean','Middle East & North Africa','North America','South Asia','Sub-Saharan Africa']:
        cleaned = data.loc[(data["Year"] == 2015) | (data["Year"] == 2016) | (data["Year"] == 2017) | (data["Year"] == 2018) | (data["Year"] == 2019)]
        coefficient, pvalue = pearsonr(cleaned[cleaned['Region'] == region]['che_gdp'], cleaned[cleaned['Region'] == region][col])
        return coefficient, pvalue
    else:
        return
# effectOfexp(MaternalMortality_clean,'MaternalMortalityRatio')


# In[25]:


#
LifeExpectancy_1 = getDataFrame('https://ghoapi.azureedge.net/api/WHOSIS_000001')
Under5_Mortality_1 = getDataFrame('https://ghoapi.azureedge.net/api/MDG_0000000007')
MaternalMortalityRatio_1 = getDataFrame('https://ghoapi.azureedge.net/api/MDG_0000000026')
global_data=getDataFrame('https://frontdoor-l4uikgap6gz3m.azurefd.net/DEX_CMS/GHE_FULL?&$orderby=VAL_DEATHS_RATE100K_NUMERIC%20desc&$select=DIM_COUNTRY_CODE,DIM_GHECAUSE_CODE,DIM_GHECAUSE_TITLE,DIM_YEAR_CODE,DIM_SEX_CODE,DIM_AGEGROUP_CODE,VAL_DALY_COUNT_NUMERIC,VAL_DEATHS_COUNT_NUMERIC,ATTR_POPULATION_NUMERIC,VAL_DALY_RATE100K_NUMERIC,VAL_DEATHS_RATE100K_NUMERIC&$filter=FLAG_RANKABLE%20eq%201%20and%20DIM_SEX_CODE%20eq%20%27BTSX%27%20and%20DIM_AGEGROUP_CODE%20eq%20%27ALLAges%27')


# In[62]:


incomegroup = pd.read_excel("datasources/income group.xlsx",sheet_name='Sheet1')
region = pd.read_csv("datasources/country_region.csv")[['Country','Region']]
healthExp = pd.read_excel("datasources/healthExp_data.xlsx",sheet_name='cleaned')
healthExp = healthExp[['country','year','che_gdp']]


# In[17]:



incomegroup = pd.melt(incomegroup, id_vars=['CountryCode','CountryName'], value_vars=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020], var_name='Year', value_name='incomeGroup')
# left join with region table.

countryInfo = incomegroup.merge(region, left_on='CountryName', right_on='Country') # inner join
countryInfo = countryInfo[['Country','CountryCode','Region','Year','incomeGroup']]
# merge datasets based on two columns:
countryInfoAll = countryInfo.merge(healthExp, left_on=['Country','Year'], right_on=['country','year']) # inner join
countryInfoAll = countryInfoAll[['Country','Region','CountryCode','Year','incomeGroup','che_gdp']]
countryInfoAll


# In[30]:


global_data_1=global_data.rename(columns={"DIM_COUNTRY_CODE": "countryCode",
                          "DIM_YEAR_CODE":"year",
                          "ATTR_POPULATION_NUMERIC":"population",
                          "DIM_GHECAUSE_CODE":"diseaseCode",
                          "DIM_GHECAUSE_TITLE":"diseaseName",
                          "VAL_DEATHS_COUNT_NUMERIC":"deathNum"
                          })

global_data_1['diseaseCode']=global_data_1['diseaseCode'].astype('int')
global_data_1['year']=global_data_1['year'].astype('int')

global_data_1 = global_data_1.merge(countryInfo, how='left',left_on=['countryCode','year'],right_on=['CountryCode','Year'] ) # left join

global_data_1['mortality_type']=None
global_data_1.loc[global_data_1['diseaseCode']<600,'mortality_type']='communicable'
global_data_1.loc[(global_data_1['diseaseCode']>600) & (global_data_1['diseaseCode']<1510),'mortality_type']='noncommunicable'
global_data_1.loc[global_data_1['diseaseCode']>1510,'mortality_type']='Injuries'


# In[31]:


global_data_1=global_data_1[['countryCode','Country','incomeGroup','year','population','mortality_type','diseaseCode','diseaseName','deathNum']]


# In[32]:


Integrity_Check(global_data_1)


# In[33]:


world_mortality_trend(global_data_1)


# In[34]:


d=ratetrend_incomegroup(global_data_1)
plotdata=d.pivot(index='year', columns='incomeGroup', values='death_per1000')
plotdata.plot(kind='line', xlabel='year',ylabel='death_per1000 ',
                 title='death rate trend from 2000-2019 by income_group',
                 figsize=(12, 8),legend=True,xticks=(range(2000,2020,1)))


# In[35]:


deathtypetrend(global_data_1)


# In[36]:



print('2004')
Topdeathtype_incomegroup_year(global_data_1,2004)

'''there death types, it matched the paper1:
1. When in 2004, Communicable diseases remain an important cause of death in lowincome countries.
2. Confirm the growing importance of noncommunicable diseases in most low- and middle-income countries.'''


# In[37]:



deathtypetrend(global_data_1,income_group='L')
deathtypetrend(global_data_1,income_group='LM')
deathtypetrend(global_data_1,income_group='UM')
deathtypetrend(global_data_1,income_group='H')


# In[38]:


print('\nGlobally leading specific disease\n\n',Topcause_year(global_data_1,2004))

print('\n\nLow-income leading specific disease\n\n ',Topcause_year(global_data_1,2004,income_group='L'))
print('\n\nLow-middle income leading specific disease\n\n',Topcause_year(global_data_1,2004,income_group='LM'))
print('\n\nUpper-middle income leading specific disease\n\n',Topcause_year(global_data_1,2004,income_group='UM'))
print('\n\nHigh income leading specific disease\n\n',Topcause_year(global_data_1,2004,income_group='H'))


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


Topcause_trend(global_data_1,2004,income_group='L',category='communicable')


# In[40]:


Topcause_trend(global_data_1,2005,income_group='UM',category='noncommunicable')


# In[41]:


print(Topcause_year(global_data_1,2004))
print(Topcause_year(global_data_1,2004,income_group='L',category='noncommunicable'))


# In[42]:


lineplot_time(LifeExpectancy_1)


# In[43]:


# Sub-Saharan Africa is part of African Region AFRO.
lineplot_time(Under5_Mortality_1)


# In[44]:


Under5_Mortality_1[(Under5_Mortality_1['TimeDim'] == 2010) & (Under5_Mortality_1['SpatialDimType'] == "COUNTRY")& (Under5_Mortality_1['Dim1'] == "BTSX")][['NumericValue','SpatialDim']]


# In[45]:


# 2010: country mortality
countryInfoAll[countryInfoAll['CountryCode'] == 'HTI']


# In[46]:



Under5_Mortality = formatWithSex('Under5_Mortality', Under5_Mortality_1)
LifeExpectancy = formatWithSex('LifeExpectancy', LifeExpectancy_1)
Under5_Mortality


# In[47]:


MaternalMortalityRatio = formatWithoutSex(MaternalMortalityRatio_1)


# In[48]:


MaternalMortalityRatio = MaternalMortalityRatio.sort_values('year')
sns.lineplot(data=MaternalMortalityRatio, x="year", sort=False,y="MaternalMortalityRatio")


# In[49]:


InfoAll = countryInfoAll.merge(LifeExpectancy, left_on=['CountryCode','Year'], right_on=['countrycode','year'], how= 'left')
InfoAll = InfoAll.merge(Under5_Mortality, left_on=['CountryCode','Year'], right_on=['countrycode','year'], how= 'left')
InfoAll = InfoAll.merge(MaternalMortalityRatio, left_on=['CountryCode','Year'], right_on=['countrycode','year'], how= 'left')
InfoAll = InfoAll[['Country','Region','CountryCode','Year','incomeGroup','che_gdp','LifeExpectancy_BTSX','LifeExpectancy_FMLE','LifeExpectancy_MLE','Under5_Mortality_BTSX','Under5_Mortality_FMLE','Under5_Mortality_MLE','MaternalMortalityRatio']]


# ### all data includes health expendure, three factors, year, incomegroup, and region for each country

# ###### anova table shows that all three factors were related to health expenditure, of which the region has the most significant correlation

# In[50]:


model = ols('che_gdp ~ Year+incomeGroup+Region', data=InfoAll).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table


# In[51]:


yearexp = InfoAll[['che_gdp','Year']].dropna()
pearsonr(yearexp['che_gdp'], yearexp['Year'])


# In[52]:


InfoAll2015_ = InfoAll.loc[(InfoAll["Year"] == 2015) | (InfoAll["Year"] == 2016) | (InfoAll["Year"] == 2017) | (InfoAll["Year"] == 2018) | (InfoAll["Year"] == 2019)]
d = InfoAll2015_[['Region','Year','che_gdp']]
d = d.dropna()
d = d.pivot_table(index='Region', columns='Year',values = 'che_gdp', aggfunc='mean')
d


# ###### correlation between independent var = region and dependent var = che_gdp in Year 2019

# In[53]:


# CAN is only country in North America
ax = sns.boxplot(x='Region', y='che_gdp', data= InfoAll[InfoAll['Year'] == 2019], color='#99c2a2')
# sns.set(rc={'figure.figsize':(16,11)})
ax = sns.swarmplot(x="Region", y="che_gdp", data=  InfoAll[InfoAll['Year'] == 2019],color=".2")
ax.set_xticklabels(ax.get_xticklabels(),rotation=30, ha="right")

plt.show()
# highest value: Tuvalu in East Asia


# ##### Effect of Health Expenditure on Selected Health Outcomes

# In[59]:


lifeExp_clean = InfoAll[['Country','Region','CountryCode','Year','incomeGroup','che_gdp', 'LifeExpectancy_BTSX']].dropna()
#
Under5_clean = InfoAll[['Country','Region','CountryCode','Year','incomeGroup','che_gdp', 'Under5_Mortality_BTSX']].dropna()
MaternalMortality_clean = InfoAll[['Country','Region','CountryCode','Year','incomeGroup','che_gdp', 'MaternalMortalityRatio']].dropna()


# In[55]:


print(effectOfexp2015before(Under5_clean,'Under5_Mortality_BTSX','Sub-Saharan Africa'))
print(effectOfexp2015before(Under5_clean,'Under5_Mortality_BTSX','Latin America & Caribbean')) #
print(effectOfexp2015before(Under5_clean,'Under5_Mortality_BTSX','Europe & Central Asia')) #
print(effectOfexp2015before(Under5_clean,'Under5_Mortality_BTSX','Middle East & North Africa')) #
print(effectOfexp2015before(Under5_clean,'Under5_Mortality_BTSX','North America')) #
print(effectOfexp2015before(Under5_clean,'Under5_Mortality_BTSX','South Asia'))
print(effectOfexp2015before(Under5_clean,'Under5_Mortality_BTSX','East Asia & Pacific'))


# In[58]:


print(effectOfexp2015after(Under5_clean,'Under5_Mortality_BTSX','Sub-Saharan Africa'))
print(effectOfexp2015after(Under5_clean,'Under5_Mortality_BTSX','Latin America & Caribbean')) #
print(effectOfexp2015after(Under5_clean,'Under5_Mortality_BTSX','Europe & Central Asia'))
print(effectOfexp2015after(Under5_clean,'Under5_Mortality_BTSX','Middle East & North Africa'))
print(effectOfexp2015after(Under5_clean,'Under5_Mortality_BTSX','North America'))
print(effectOfexp2015after(Under5_clean,'Under5_Mortality_BTSX','South Asia'))
print(effectOfexp2015after(Under5_clean,'Under5_Mortality_BTSX','East Asia & Pacific'))


# In[56]:


print(effectOfexp2015after(lifeExp_clean,'LifeExpectancy_BTSX','Sub-Saharan Africa'))
print(effectOfexp2015after(lifeExp_clean,'LifeExpectancy_BTSX','Latin America & Caribbean')) #
print(effectOfexp2015after(lifeExp_clean,'LifeExpectancy_BTSX','Europe & Central Asia'))
print(effectOfexp2015after(lifeExp_clean,'LifeExpectancy_BTSX','Middle East & North Africa'))
print(effectOfexp2015after(lifeExp_clean,'LifeExpectancy_BTSX','North America'))
print(effectOfexp2015after(lifeExp_clean,'LifeExpectancy_BTSX','South Asia'))
print(effectOfexp2015after(lifeExp_clean,'LifeExpectancy_BTSX','East Asia & Pacific'))


# In[60]:


#todo need to fix the bug: ValueError: x and y must have length at least 2.
print(effectOfexp2015after(MaternalMortality_clean,'MaternalMortalityRatio','Sub-Saharan Africa'))
print(effectOfexp2015after(MaternalMortality_clean,'MaternalMortalityRatio','Latin America & Caribbean')) #
print(effectOfexp2015after(MaternalMortality_clean,'MaternalMortalityRatio','Europe & Central Asia'))
print(effectOfexp2015after(MaternalMortality_clean,'MaternalMortalityRatio','Middle East & North Africa'))
print(effectOfexp2015after(MaternalMortality_clean,'MaternalMortalityRatio','North America'))
print(effectOfexp2015after(MaternalMortality_clean,'MaternalMortalityRatio','South Asia'))
print(effectOfexp2015after(MaternalMortality_clean,'MaternalMortalityRatio','East Asia & Pacific'))


# In[ ]:




