

# Historical changes in health conditions and their relation to government expenditure

Team members：      
Yanying Yu/ ABELAAAZ  
Shufan Ming/ Michelle-Mings

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

The Impact of Health Expenditures on Health Outcomes in Sub-Saharan Africa: https://journals.sagepub.com/doi/pdf/10.1177/0169796X19826759
Global and regional causes of death: https://pubmed.ncbi.nlm.nih.gov/19776034/

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


# Conclusions 

### H1

### The effect of Health Expenditure on Selected Health Outcomes
**1**. Steady increases in health expenditures over time have the tendency to improve health outcomes in most regions/countries. Both under-five mortality and maternal mortality show a decreasing trend, meanwhile the average life expectancy is increasing (with the exception in Sub-Saharan Africa and South Asia). 
**2**. After 2015, health expenditure continue to exert a positive impact on all three health outcomes. 

# Short description of how to run the code
**1**.Firstly, please clone this repository into your local machine  
  **OR**  
You can download ’Datasources’ and the files under this directory &  IS597 Health and death.ipynb & IS597 Health and death.py. Make sure the path of each file is correct since we use the relative path.   
    
**2**. Run analysis_visualization.ipynb    

**3**. To check the doctest and doctest with coverage, please see the functions.py file


