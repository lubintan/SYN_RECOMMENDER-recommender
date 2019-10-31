
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
GENDER_GROUP_LIST = ['Male', 'Female']
LIFESTAGE_GROUP_LIST = ['Single', 'Married','Married with dependents']
AGE_GROUP_LIST = ['20s', '30s', '40s', '50s']
INCOME_GROUP_LIST = ['<50k', '> 50K', '> 150K']

def read(filename):
    df = pd.read_excel(filename)

    discardedColumns = ['S/NO','POLICY STATUS','BASIC PLAN', 'PROVIDER','NO OF ENTRIES','Unamed: 55', 'Unnamed: 56', 'Unnamed: 57']

    for col in discardedColumns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    newDf = df.groupby(by=CLIENT_LABEL).sum()

    newDf = np.clip(newDf,a_max=2,a_min=None)
    newDf.index.name = CLIENT_LABEL


    df = df[[CLIENT_LABEL,GENDER_LABEL,AGE_GROUP_LABEL,INCOME_LABEL,LIFESTAGE_LABEL]]
    df = df.drop_duplicates(subset=CLIENT_LABEL)
    df = df.sort_values(by=CLIENT_LABEL)


    newDf = pd.merge(left=df, right=newDf, on=CLIENT_LABEL)

    return newDf



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

                    thisDf.to_csv('tables_72_DE//' + str(counter) +'_' + gender + '_' + ageGroup + '_' +
                                  lifeStage + '_' + incomeLvl+'.csv', index=False)




if __name__ == '__main__':

    # Generate 4 groupings
    start = time.time()
    df = read('Domain Expert Training Data D2.xlsx')
    df.to_csv('output_DE.csv', index=False)

    end = time.time()
    print('Time taken: %.2f seconds' % (end-start))

    # Generate 72 tables
    start = time.time()

    gen72tables(df)

    end = time.time()
    print('Time taken: %.2f seconds' % (end - start))






