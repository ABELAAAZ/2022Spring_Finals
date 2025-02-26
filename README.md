
 
# Historical changes in health conditions and their relation to government expenditure
<img src="https://datascienceethics.com/wp-content/uploads/2019/04/99314779_s.jpg" style="width:500px" /> <br/>   
Team members：      
Yanying Yu/ ABELAAAZ  
Shufan Ming/ Michelle-Mings    

# Short description of how to run the code
**1**.Firstly, please clone this repository into your local machine  
  **OR**  
You can download ’Datasources’ and the files under this directory &  functions.py & analysis_visualization.ipynb. Make sure the path of each file is correct since we use the relative path.   
    
**2**. Run analysis_visualization.ipynb    

**3**. To check the doctest and doctest with coverage, please see the functions.py file


# Assumptions 
**Seven regions**  — Provided by The World Bank. Assume not modified.    
East Asia & Pacific; Europe & Central Asia; Latin America & Caribbean;    
Middle East & North Africa; Middle East & North Africa; South Asia; Sub-Saharan Africa    
        
**Four income group** — Provided by The World Bank, changes every year. Adopt using the classification of that year.   
1.High  2.Upper-middle  3.Lower-middle  4.Low      
	
**Three types of Mortality**  — Provided by World Health Organization (WHO). Assume not modified.      
   
1.communicable 2.noncommunicable 3.injuries.       
			
**Diseases code** — provided by WHO Global Health Estimates (GHE). Assume not modified.      

# Short description of the problem solved
The big topic is to observe the trend of health-related issues and , from the perspective of life expectancy, mortality in different age groups, and cause of death and its relations to income groups, regions, and health expenditures. Our work is based on two previous publications which are listed below:

Global and regional causes of death:    
https://pubmed.ncbi.nlm.nih.gov/19776034/         
The Impact of Health Expenditures on Health Outcomes in Sub-Saharan Africa:    
https://journals.sagepub.com/doi/pdf/10.1177/0169796X19826759         
  

Those two papers have more or less limitations: data is outdated. No clear data pre-processing explanations, So one of our purpose is to replicate the analysis in the paper to see whether we reach an agreement on their conclusions; and also whether the finding for one region, that patterns can be generalised to other regions as well    

## Data sources:
1. Global cause of death   
API:https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/ghe-leading-causes-of-death      
2. Health expenditure:     
API: https://www.who.int/data/gho/info/gho-odata-api    
3. Country income group, Country Region, Mortality categories,    
https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups    
https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/ghe-leading-causes-of-death    
https://datatopics.worldbank.org/world-development-indicators/the-world-by-income-and-region.html    

## Hypothesis
H1:Do the Global patterns of mortality by income group all over the world aligned with the previous work? (Patterns: i.e., Trend of mortality/ Cause Distribution/ Leading cause of death

H2:As in Sub-Saharan Africa, we expect health expenditure to exert a positive and significant impact on all three health outcomes (life expectancy, under-five mortality, and maternal mortality), for other regions globally. (1996-2015)


# Conclusions<br/>  (Please see the details in analysis_visualization.ipynb and 597Final-presentation.pdf)         
 
## The Global patterns of mortality all over the world match the conclusion of the paper 1 to some extents                     		
**1.Trend of mortality:**<br/>			
**1.1**True: There is a growing importance of non-communicable disease all over the world.<br/>
**1.2**False:There is a growing importance of noncommunicable diseases in most low- and middle-income countries since 2004. <br/>	            		
**2.Cause Distribution:**<br/>		
**2.1**True: The contribution of three mortality types in 2004 is noncommunicable 60%, communicable 30%, injuries 10%.<br/>	          		
**3.leading cause of death:**<br/>
**3.1**True:Cardiovascular diseases are the leading cause of death,Infectious and parasitic diseases are the next leading cause, followed by cancers in 2004 all over the world.		         
**3.2**False: Same top 10 leading causes-of-deaths in the specific income group.<br/>	    

#### Other findings:    
1 Low income countries did a great job on communicable disease issue.    
2. There are major differences in the ranking of causes between high- and low-income countries    
3. An HIV pandemic occurred in 2007 in low-income countries with 7 years to come back to the normal level.    
4. IHD and Stroke cases surged in 2004 in upper-middle countries, leading to a mortality spike in the whole Upper-middle income countries    


## The effect of Health Expenditure on Selected Health Outcomes
**1. The in Sub-Saharan Africa, before 2015, 1 percent increase in health expenditure per capita improve life expectancy by 0.06 percent. Our results match this conclusion of the paper 2.**<br/>
**2. After 2015, steady increases in health expenditures over time have the tendency to improve all three health outcomes in most regions/countries, with the exception in Sub-Saharan Africa, East Asia & Pacific, and South Asia.**<br/>    
**2.1**. After 2015, health expenditure continue to exert a positive impact on under-five mortality and maternal mortality all over the world, they both show decreasing trends, except for Sub-Saharan Africa and South Asia.<br/>
**2.2**. Meanwhile, the average life expectancy is increasing, except for Sub-Saharan Africa, East Asia & Pacific, and South Asia.<br/>

Other details please see the markdown cells in analysis_visualization.ipynb
