# Recommender System

## `recommender.py`
This uses the Surprise library (surprise.readthedocs.io), and evaluates recommending algorithms as well as gives Top-N Recommendations for a specified user.

### `dataProcessor.py`

### Empty Age Info
Empty AGE rows are filled in using numbers in the existing data to extend the data such that the
existing distribution percentages remain the same.

### Inferring Groupings
The 3 inferred groupings, Gender, Lifestage and Income Level, are inferred using MOM's June 2018 data
on the Singapore labourforce from age 15 and up, found at
https://stats.mom.gov.sg/Pages/Labour-Force-In-Singapore-2018.aspx.

### Data Sources
In particular, data found in tables 31 and 72, found at the following link, were used.
https://stats.mom.gov.sg/Pages/Labour-Force-in-Singapore-2018-Employment.aspx

Using the gender ratio for each age group, the user entries were first assigned genders.
Eg. For those in their 30s, if the gender ratio is 40:60 male-female, then of the user entries in the 30s age group,
40% are assigned 'M' and the rest are assigned 'F'.

Then using the lifestage percentages for each gender per age group, the user entries were assigned lifestages.
Eg. Of the males in their 30s, 40% are single, 40% are married and 20% are widowed or divorced.
Then of the user entries in the 30s age group who are male, 40% are assigned 'single', 40% are assigned 'married' and
20% are assigned 'widowed/divorced'.

The income level assignments are similar to the lifestage assignments.
Using the percentages for each gender per age group, the user entries were assigned income level brackets.
Eg. Of the males in their 30s, 20% earn below 2k monthly, 20% earn between 2-6k, 30% earn between 6-10k,
and 30% earn above 10k.
Then of the user entries in the 30s age group who are male, 20% are assigned 'Below 2,000',
20% are assigned '2,000 - 5,999', 30% are assigned '6,000 - 9,999', and 30% are assigned '10,000 and above'.

The excel data from MOM and the preliminary consolidation and calculations to get the required ratios and percentages
for this script to run can be found in the `Data_Sources` folder.

