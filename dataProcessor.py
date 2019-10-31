'''
Empty AGE rows are filled in using numbers in the existing data to extend the data such that the
existing distribution percentages remain the same.

The 3 inferred groupings, Gender, Lifestage and Income Level, are inferred using MOM's June 2018 data
on the Singapore labourforce from age 15 and up, found at
https://stats.mom.gov.sg/Pages/Labour-Force-In-Singapore-2018.aspx.

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
for this script to run can be found in the 'Data_Sources' folder.

'''

import pandas as pd
import numpy as np
import time


### TODO:
# 1. Combine ratings of similar users.
# 2. Extend age information for those without age information.
# 3. Split the data by age group. Then:
# 4. Gender: Male-Female
# 5. Lifestage
# 6. Income Level


CLIENT_LABEL = 'CLIENTS'
AGE_LABEL = 'AGE'
AGE_GROUP_LABEL = 'AGE GROUP'
GENDER_LABEL = 'GENDER'
LIFESTAGE_LABEL = 'LIFESTAGE'
INCOME_LABEL = 'INCOME GROUP'
GENDER_GROUP_LIST = ['M', 'F']
LIFESTAGE_GROUP_LIST = ['Single', 'Married','Widowed_Divorced']
AGE_GROUP_LIST = ['20s and below', '30s', '40s', '50s and above']
INCOME_GROUP_LIST = ['Below 4000', '4000 - 9999', '10000 and above']

def readAndAggregate(filename):
    df = pd.read_csv(filename)

    discardedColumns = ['S/NO','POLICY STATUS','BASIC PLAN', 'PROVIDER','NO OF ENTRIES', 'Unnamed: 56', 'Unnamed: 57']

    for col in discardedColumns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    newDf = df.groupby(by=CLIENT_LABEL).sum()

    newDf = np.clip(newDf,a_max=2,a_min=None)
    newDf.index.name = CLIENT_LABEL

    # For getting Age from DOB
    # newDf = getAgeFromDOB(df, newDf)

    df = df[[CLIENT_LABEL, AGE_LABEL]]
    df = df.drop_duplicates(subset=CLIENT_LABEL)
    df[AGE_LABEL] = pd.to_numeric(df[AGE_LABEL], errors='coerce')
    # df = df.sort_values(by=CLIENT_LABEL)

    df = updateAgeDistributionInData(df)

    newDf = pd.merge(left=df, right=newDf, on=CLIENT_LABEL)

    return newDf

def getAgeFromDOB(originalDf, newDf):


    originalDf['age'] = np.floor((pd.to_datetime('today') - pd.to_datetime(originalDf.DOB)).dt.days/365.25)
    originalDf = originalDf.drop_duplicates(subset=CLIENT_LABEL)
    originalDf = originalDf[[CLIENT_LABEL,'age']]

    newDf = pd.merge(left=originalDf, right=newDf, on=[CLIENT_LABEL])

    print(newDf)

    return newDf


def updateAgeDistributionInData(inputDf):

    df = inputDf[[CLIENT_LABEL, AGE_LABEL]]

    total = len(df)

    ageUnder30s = len(df[df[AGE_LABEL] < 30])
    age30s = len(df[(df[AGE_LABEL] >= 30) & (df[AGE_LABEL] < 40)])
    age40s = len(df[(df[AGE_LABEL] >= 40) & (df[AGE_LABEL] < 50)])
    ageAbove40s = len(df[df[AGE_LABEL] >= 50])

    ageNaN = df.AGE.isna().sum()

    sum = ageUnder30s + age30s + age40s + ageAbove40s


    percentUnder30s = float(ageUnder30s)/ sum
    percent30s = float(age30s)/sum
    percent40s = float(age40s) / sum

    ageDf = df[df.AGE.isna()]
    ageDf = ageDf.reset_index(drop=True)

    numUnder30s = int(percentUnder30s*total - ageUnder30s)
    num30s = int(percent30s*total - age30s)
    num40s = int(percent40s*total - age40s)
    numAbove40s = ageNaN - numUnder30s - num30s - num40s

    for i, r in ageDf.iterrows():
        if numUnder30s > 0:
            inputDf.loc[inputDf[CLIENT_LABEL] == r.CLIENTS, AGE_LABEL] = 25
            numUnder30s -= 1
        elif num30s > 0:
            inputDf.loc[inputDf[CLIENT_LABEL] == r.CLIENTS, AGE_LABEL] = 35
            num30s -= 1
        elif num40s > 0:
            inputDf.loc[inputDf[CLIENT_LABEL] == r.CLIENTS, AGE_LABEL] = 45
            num40s -= 1
        elif numAbove40s > 0:
            inputDf.loc[inputDf[CLIENT_LABEL] == r.CLIENTS, AGE_LABEL] = 55
            numAbove40s -= 1


    return inputDf

def filterByAgeGroups(df):
    df_under30s = pd.DataFrame(df[df[AGE_LABEL] < 30])
    df_30s = pd.DataFrame(df[(df[AGE_LABEL] >= 30) & (df[AGE_LABEL] < 40)])
    df_40s = pd.DataFrame(df[(df[AGE_LABEL] >= 40) & (df[AGE_LABEL] < 50)])
    df_above40s = pd.DataFrame(df[df[AGE_LABEL] >= 50])

    return df_under30s, df_30s, df_40s, df_above40s

def groupBy4Factors(inputDf):



    df = inputDf[[CLIENT_LABEL,AGE_LABEL]]
    df_under30s, df_30s, df_40s, df_above40s = filterByAgeGroups(df)

    df_under30s, df_30s, df_40s, df_above40s = groupByAgeGroups(df_under30s, df_30s, df_40s, df_above40s)
    df_under30s, df_30s, df_40s, df_above40s = groupByGender(df_under30s, df_30s, df_40s, df_above40s)
    df_under30s, df_30s, df_40s, df_above40s = groupByLifestage(df_under30s, df_30s, df_40s, df_above40s)
    df_under30s, df_30s, df_40s, df_above40s = groupByIncomeLevel(df_under30s, df_30s, df_40s, df_above40s)

    df = pd.concat([df_under30s, df_30s, df_40s, df_above40s], axis = 0)

    df = df[[CLIENT_LABEL,
             GENDER_LABEL,
             LIFESTAGE_LABEL,
             INCOME_LABEL,
             AGE_GROUP_LABEL,
             ]]

    inputDf = pd.merge(left = df, right = inputDf, on = CLIENT_LABEL)
    inputDf = inputDf.sort_values(by=CLIENT_LABEL)

    return inputDf

def groupByAgeGroups(df_under30s, df_30s, df_40s, df_above40s):

    print('Grouping by Age Groups..')

    df_under30s[AGE_GROUP_LABEL] = AGE_GROUP_LIST[0]
    df_30s[AGE_GROUP_LABEL] = AGE_GROUP_LIST[1]
    df_40s[AGE_GROUP_LABEL] = AGE_GROUP_LIST[2]
    df_above40s[AGE_GROUP_LABEL] = AGE_GROUP_LIST[3]

    return df_under30s, df_30s, df_40s, df_above40s




def groupByGender(df_under30s, df_30s, df_40s, df_above40s):

    print('Grouping  by Gender..')

    distDict = {}
    distDict[AGE_GROUP_LIST[0]]= {'M': 0.558723651, 'F': 0.442220636}
    distDict[AGE_GROUP_LIST[1]] = {'M': 0.510307088, 'F': 0.489692912}
    distDict[AGE_GROUP_LIST[2]] = {'M': 0.531742345, 'F': 0.468257655}
    distDict[AGE_GROUP_LIST[3]] = {'M': 0.593723944, 'F': 0.406055305}

    inputDfList = [df_under30s, df_30s, df_40s, df_above40s]

    for n in range(len(inputDfList)):
        total = len(inputDfList[n])
        numMales = int(distDict[AGE_GROUP_LIST[n]]['M'] * total)
        numFemales = total - numMales

        thisDf = inputDfList[n]

        for i, r in thisDf.iterrows():
            if numMales > 0:
                thisDf.loc[thisDf[CLIENT_LABEL] == r.CLIENTS, GENDER_LABEL] = 'M'
                numMales -= 1
            elif numFemales > 0:
                thisDf.loc[thisDf[CLIENT_LABEL] == r.CLIENTS, GENDER_LABEL] = 'F'
                numFemales -= 1

        inputDfList[n] = thisDf


    return inputDfList[0], inputDfList[1], inputDfList[2], inputDfList[3]


def groupByLifestage(df_under30s, df_30s, df_40s, df_above40s):

    print('Grouping by Lifestage..')

    inputDfList = [df_under30s, df_30s, df_40s, df_above40s]

    distDict = {}
    distDict[AGE_GROUP_LIST[0]] = [
    0.914292463, # 0 - Single Male
    0.872629682, # 1 - Single Female
    0.083294264, # 2 - Married Male
    0.122587019, # 3 - Married Female
    0.002883419, # 4 - Widowed/Divorced Male
    0.004270479, # 5 - Widowed/Divorced Female
    ]

    distDict[AGE_GROUP_LIST[1]] = [
    0.284402834,
    0.244646854,
    0.702756917,
    0.71046678,
    0.012420785,
    0.044886367,
    ]

    distDict[AGE_GROUP_LIST[2]] = [
    0.130028764,
    0.159716213,
    0.824292003,
    0.740499201,
    0.045679234,
    0.100193751,
    ]

    distDict[AGE_GROUP_LIST[3]] = [
    0.073154273,
    0.152749807,
    0.858794965,
    0.635327175,
    0.068050763,
    0.211923018,
    ]

    for n in range(len(inputDfList)):
        thisDf = inputDfList[n]

        maleDf = pd.DataFrame(thisDf[thisDf[GENDER_LABEL] == 'M'])
        femaleDf = pd.DataFrame(thisDf[thisDf[GENDER_LABEL] == 'F'])

        genderedDfs = [maleDf, femaleDf]

        for genN in range(len(genderedDfs)):

            total = len(genderedDfs[genN])

            numSingles = int(distDict[AGE_GROUP_LIST[n]][genN%2] * total)
            numMarrieds = int(distDict[AGE_GROUP_LIST[n]][(genN%2) + 2] * total)
            numWidowed = total - numSingles - numMarrieds

            for i, r in genderedDfs[genN].iterrows():
                if numSingles > 0:
                    genderedDfs[genN].loc[genderedDfs[genN][CLIENT_LABEL] == r.CLIENTS, LIFESTAGE_LABEL] = LIFESTAGE_GROUP_LIST[0]
                    numSingles -= 1
                elif numMarrieds > 0:
                    genderedDfs[genN].loc[genderedDfs[genN][CLIENT_LABEL] == r.CLIENTS, LIFESTAGE_LABEL] = LIFESTAGE_GROUP_LIST[1]
                    numMarrieds -= 1
                elif numWidowed > 0:
                    genderedDfs[genN].loc[genderedDfs[genN][CLIENT_LABEL] == r.CLIENTS, LIFESTAGE_LABEL] = LIFESTAGE_GROUP_LIST[2]
                    numWidowed -= 1

        inputDfList[n] = pd.concat(genderedDfs, axis=0)

    return inputDfList[0], inputDfList[1], inputDfList[2], inputDfList[3]

def groupByIncomeLevel(df_under30s, df_30s, df_40s, df_above40s):
    print('Grouping by Income Level..')

    inputDfList = [df_under30s, df_30s, df_40s, df_above40s]

    distMale = {}
    distMale[AGE_GROUP_LIST[0]] = {}
    distMale[AGE_GROUP_LIST[1]] = {}
    distMale[AGE_GROUP_LIST[2]] = {}
    distMale[AGE_GROUP_LIST[3]] = {}

    distMale[AGE_GROUP_LIST[0]][INCOME_GROUP_LIST[0]] = 0.76795
    distMale[AGE_GROUP_LIST[1]][INCOME_GROUP_LIST[0]] = 0.36213
    distMale[AGE_GROUP_LIST[2]][INCOME_GROUP_LIST[0]] = 0.34615
    distMale[AGE_GROUP_LIST[3]][INCOME_GROUP_LIST[0]] = 0.62613

    distMale[AGE_GROUP_LIST[0]][INCOME_GROUP_LIST[1]] = 0.21923
    distMale[AGE_GROUP_LIST[1]][INCOME_GROUP_LIST[1]] = 0.49112
    distMale[AGE_GROUP_LIST[2]][INCOME_GROUP_LIST[1]] = 0.41108
    distMale[AGE_GROUP_LIST[3]][INCOME_GROUP_LIST[1]] = 0.24149

    distMale[AGE_GROUP_LIST[0]][INCOME_GROUP_LIST[2]] = 0.01282
    distMale[AGE_GROUP_LIST[1]][INCOME_GROUP_LIST[2]] = 0.14675
    distMale[AGE_GROUP_LIST[2]][INCOME_GROUP_LIST[2]] = 0.24277
    distMale[AGE_GROUP_LIST[3]][INCOME_GROUP_LIST[2]] = 0.13238

    distFemale = {}
    distFemale[AGE_GROUP_LIST[0]] = {}
    distFemale[AGE_GROUP_LIST[1]] = {}
    distFemale[AGE_GROUP_LIST[2]] = {}
    distFemale[AGE_GROUP_LIST[3]] = {}

    distFemale[AGE_GROUP_LIST[0]][INCOME_GROUP_LIST[0]] = 0.77199
    distFemale[AGE_GROUP_LIST[1]][INCOME_GROUP_LIST[0]] = 0.45234
    distFemale[AGE_GROUP_LIST[2]][INCOME_GROUP_LIST[0]] = 0.48176
    distFemale[AGE_GROUP_LIST[3]][INCOME_GROUP_LIST[0]] = 0.74066

    distFemale[AGE_GROUP_LIST[0]][INCOME_GROUP_LIST[1]] = 0.21959
    distFemale[AGE_GROUP_LIST[1]][INCOME_GROUP_LIST[1]] = 0.46713
    distFemale[AGE_GROUP_LIST[2]][INCOME_GROUP_LIST[1]] = 0.36513
    distFemale[AGE_GROUP_LIST[3]][INCOME_GROUP_LIST[1]] = 0.19697

    distFemale[AGE_GROUP_LIST[0]][INCOME_GROUP_LIST[2]] = 0.00843
    distFemale[AGE_GROUP_LIST[1]][INCOME_GROUP_LIST[2]] = 0.08053
    distFemale[AGE_GROUP_LIST[2]][INCOME_GROUP_LIST[2]] = 0.15311
    distFemale[AGE_GROUP_LIST[3]][INCOME_GROUP_LIST[2]] = 0.06236

    genderedDistDicts = [distMale, distFemale]
    
    for n in range(len(inputDfList)):
        thisDf = inputDfList[n]

        maleDf = pd.DataFrame(thisDf[thisDf[GENDER_LABEL] == 'M'])
        femaleDf = pd.DataFrame(thisDf[thisDf[GENDER_LABEL] == 'F'])

        genderedDfs = [maleDf, femaleDf]

        for genN in range(len(genderedDfs)):

            total = len(genderedDfs[genN])

            num4kBelow = int(genderedDistDicts[genN][AGE_GROUP_LIST[n]][INCOME_GROUP_LIST[0]] * total)
            num4k_10k = int(genderedDistDicts[genN][AGE_GROUP_LIST[n]][INCOME_GROUP_LIST[1]] * total)
            numAbove10k = total - num4kBelow - num4k_10k

            for i, r in genderedDfs[genN].iterrows():
                if num4kBelow > 0:
                    genderedDfs[genN].loc[genderedDfs[genN][CLIENT_LABEL] == r.CLIENTS, INCOME_LABEL] = INCOME_GROUP_LIST[0]
                    num4kBelow -= 1
                elif num4k_10k > 0:
                    genderedDfs[genN].loc[genderedDfs[genN][CLIENT_LABEL] == r.CLIENTS, INCOME_LABEL] = INCOME_GROUP_LIST[1]
                    num4k_10k -= 1
                elif numAbove10k > 0:
                    genderedDfs[genN].loc[genderedDfs[genN][CLIENT_LABEL] == r.CLIENTS, INCOME_LABEL] = INCOME_GROUP_LIST[2]
                    numAbove10k -= 1

        inputDfList[n] = pd.concat(genderedDfs, axis=0)

    return inputDfList[0], inputDfList[1], inputDfList[2], inputDfList[3]

def gen72tables(df):

    print("Total Entries:", len(df))

    summer = 0
    counter = 0

    for gender in GENDER_GROUP_LIST:
        for ageGroup in AGE_GROUP_LIST:
            for lifeStage in LIFESTAGE_GROUP_LIST:
                for incomeLvl in INCOME_GROUP_LIST:
                    counter += 1

                    thisDf = df[df[GENDER_LABEL] == gender]
                    thisDf = thisDf[thisDf[AGE_GROUP_LABEL] == ageGroup]
                    thisDf = thisDf[thisDf[LIFESTAGE_LABEL] == lifeStage]
                    thisDf = thisDf[thisDf[INCOME_LABEL] == incomeLvl]

                    entries = len(thisDf)
                    summer += entries

                    print(counter, ",gender:", gender, ',age group:', ageGroup, ',life stage:',
                          lifeStage, ',income level:', incomeLvl, ',entries:', entries)

                    thisDf.to_csv('tables_72//' + str(counter) +'_' + gender + '_' + ageGroup + '_' +
                                  lifeStage + '_' + incomeLvl+'.csv', index=False)




if __name__ == '__main__':

    # Generate 4 groupings
    start = time.time()
    df = readAndAggregate('Raw Data.csv')
    df = groupBy4Factors(df)
    df.to_csv('output.csv', index=False)

    end = time.time()
    print('Time taken: %.2f seconds' % (end-start))

    # Generate 72 tables
    start = time.time()

    gen72tables(df)

    end = time.time()
    print('Time taken: %.2f seconds' % (end - start))






