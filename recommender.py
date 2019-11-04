import pandas as pd
import plotly.graph_objs as go
import plotly
from surprise import Reader, Dataset, KNNWithMeans, SlopeOne,SVD, \
    NMF, NormalPredictor, BaselineOnly, CoClustering, KNNBaseline, KNNBasic, KNNWithZScore,SVDpp
import surprise.model_selection as models
import random, time
import numpy as np
from os import listdir
from os.path import isfile, join
import PySimpleGUI as sg


# Parameters

SIM_LIST = [
            {"name": "pearson", "user_based": True, "min_support": 1},
            {"name": "msd", "user_based": True, "min_support": 1},
            {"name": "cosine", "user_based": True, "min_support": 1}
            ]


ALGO_LIST = [
            {"name": 'SlopeOne', "algo": SlopeOne()},
            {"name": 'KNN_Means', "algo": KNNWithMeans(verbose=False)},
            {"name": 'SVD', "algo": SVD(verbose=False)},
            ]

GENDER_GROUP_LIST = ['Male', 'Female']
LIFESTAGE_GROUP_LIST = ['Single', 'Married','Married with dependents']
AGE_GROUP_LIST = ['20s', '30s', '40s', '50s']
INCOME_GROUP_LIST = ['<50k', '> 50K', '> 150K']

PRODUCT_LIST = None
    # [
    #     'Retirement - RP with Maturity',
    #     'Retirement - RP without Maturity',
    #     'Retirement - RP with Life payout',
    #     'Retirement - SP',
    #     'RP Anticipated Endowment (Regular Pay)',
    #     'RP Anticipated Endowment (Very Short PPT)',
    #     'RP Anticipated Endowment (Short PPT)',
    #     'RP Anticipated Endowment Plans (Mid PPT)',
    #     'RP Anticipated Endowment (Long PPT)',
    #     'SP Anticipated Endowment',
    #     'RP Classic Endowment (Regular Pay)',
    #     'RP Classic Endowment (Very Short PPT)',
    #     'RP Classic Endowment (Short PPT)',
    #     'RP Classic Endowment (Mid PPT)',
    #     'RP Classic Endowment (Long PPT)',
    #     'SP Classic Endowment',
    #     'Education Funding Plans',
    #     'RP ILP - Accumulation',
    #     'SP ILP - Accumulation',
    #     'ILP - Protection (Face Plus)',
    #     'ILP - Protection (Level Face)',
    #     'Level Term',
    #     'Renewable Term',
    #     'Reducing Term',
    #     'Refundable Term',
    #     'Disability Income',
    #     'Other Term',
    #     'Long Term Care',
    #     'Integrated Shield',
    #     'International H&S',
    #     'Other H&S',
    #     'Natal Insurance',
    #     'Accident',
    #     'Other A&H',
    #     'Standalone Critical Illness',
    #     'Early Stage Critical Illness',
    #     'Cancer',
    #     'Gender Specific',
    #     'Elderly',
    #     'RP WL Protection (with multiplier)',
    #     'RP WL Protection (without multiplier)',
    #     'SP WL Protection  ',
    #     'RP WL Income',
    #     'SP WL Income',
    #     'Trad UL',
    #     'VUL',
    #     'Indexed UL',
    # ]

INCLUDE_NAS_IN_RMSE = False
RNG_SEED = 1
NUM_FOLDS = 5 # must be at least 2
K_RANGE = range(1,9)
RATING_SCALE = (0,2)
TOP_N = 3

def processData(inputFile, folds = NUM_FOLDS):

    if folds < 2:
        print('Not enough folds specified.')
        exit()

    print('Reading data..')
    df = pd.read_csv(inputFile)

    if len(df) < folds:
        return 0, 0, 0, False


    # If need to combine with product data.
    # df2 = pd.read_csv('tables_72/46_F_30s_Single_Below 4000')
    # df2.drop(labels=['GENDER', 'AGE GROUP', 'LIFESTAGE', 'INCOME GROUP','AGE'], inplace=True, axis=1)

    # df = pd.concat([df,df2], sort=False)


    gender = df.iloc[0]['GENDER']
    age_group = df.iloc[0]['AGE GROUP']
    lifestage = df.iloc[0]['LIFESTAGE']
    incomeLvl = df.iloc[0]['INCOME GROUP']

    df.drop(labels=['GENDER', 'AGE GROUP', 'LIFESTAGE', 'INCOME GROUP'], inplace=True, axis=1)

    # for each in (list(df.columns)):
    #     print(each)
    # exit()

    length = len(df)

    if length < folds:
        print("Data too small for folding %i times" % (folds))
        return 0, 0, 0, False

    else:

        chunkSize = int(length / folds)

        trainDfList = []
        testDfList = []

        for i in range(folds):
            head = i * chunkSize
            tail = (i + 1) * chunkSize

            testDf = df[head:tail]
            trainDf = pd.concat([df[0:head], df[tail:]])

            testDfList.append(processDfInto3Cols(testDf))
            trainDfList.append(processDfInto3Cols(trainDf))

    print('Data read complete.')
    print()
    # print('Gender:',gender, '\nAge Group:',age_group, "\nLifestage:",lifestage, '\nIncome:', incomeLvl, '\n')

    return testDfList, trainDfList, df, True

def processDfInto3Cols(df):
    products = df.columns[1:]  # remove 'CLIENTS'

    global PRODUCT_LIST
    PRODUCT_LIST = list(products)

    newList = []

    for i, r in df.iterrows():
        for label in products:
            thisEntry = []
            thisEntry.append(int(r['CLIENTS']))
            thisEntry.append(label)
            thisEntry.append(r[label])
            newList.append(thisEntry)

    processedDf = pd.DataFrame(newList, columns=['CLIENTS', 'ITEM', 'RATING'])

    # drop 0s
    processedDf = processedDf.dropna().reset_index(drop=True)

    processedDf = processedDf[processedDf['RATING'] != 0].reset_index(drop=True)

    return processedDf



def measureAccuracy(testSet, trainSet, overallDf, algo=None, sim=SIM_LIST[0], min_k = 5, max_k = 5, includeNAs = INCLUDE_NAS_IN_RMSE):
    MIN_ENTRIES = 1

    userCount = testSet[testSet['RATING'] > 0]
    itemCount = testSet[testSet['RATING'] > 0]

    userCount = userCount.groupby('CLIENTS').count()
    userCount = userCount.sort_values(['RATING'], ascending=[0])
    userCount = userCount[userCount['RATING'] >= MIN_ENTRIES]

    userIDs = list(userCount.index)

    itemCount = itemCount.groupby('ITEM').count()
    items = list(itemCount.index)

    reader = Reader()
    algo.sim_options = sim
    algo.min_k = min_k
    algo.k = max_k

    squaredErrorList = []
    pointsTested = 0


    if includeNAs: #tests against all rated items in this table.

        for client in userIDs:
            thisDf = testSet[testSet['CLIENTS'] == client].reset_index(drop=True)
            num = len(items)
            
            pointsTested += num

            # print('Client:', client, 'Client Items:', len(list(thisDf.ITEM)),'Num Items to Test:', num)

            for i in range(num):

                thisItem = items[i]

                if thisItem in list(thisDf.ITEM):
                    thisRow = thisDf[thisDf['ITEM'] == thisItem]
                    actualRating = thisRow.RATING.values[0]
                    index = thisRow.index.values[0]
                    newDf = thisDf.drop(index)
                    trainSet = pd.concat([trainSet, newDf])
                else:
                    actualRating = 0
                    trainSet = pd.concat([trainSet, thisDf])

                data = Dataset.load_from_df(trainSet, reader)
                trainingData = data.build_full_trainset()
                algo.fit(trainingData)

                prediction = algo.predict(client, items[i],verbose=False)

                predictedRating = prediction.est

                if 'actual_k' in prediction.details.keys():

                    was_imp = prediction.details['was_impossible']

                    if was_imp: continue

                    actual_k = prediction.details['actual_k']

                    if not ((actual_k < min_k) or (actual_k > max_k)):
                        squaredErr = (predictedRating - actualRating) ** 2
                        squaredErrorList.append(squaredErr)
                else:
                    # print('No actual k')
                    squaredErr = (predictedRating - actualRating) ** 2
                    squaredErrorList.append(squaredErr)


    else: # tests only against items that the user has rated. This is what we will be employing.

        for client in userIDs:
            thisDf = testSet[testSet['CLIENTS'] == client].reset_index(drop=True)

            num = len(thisDf)
            
            pointsTested += num

            print('Client:', client, 'Client Items:', num)

            for i in range(num):
                actualRating = thisDf.iloc[i].RATING
                thisItem = thisDf.iloc[i].ITEM

                newDf = thisDf.drop(thisDf.index[i])
                trainSet = pd.concat([trainSet, newDf])

                data = Dataset.load_from_df(trainSet, reader)
                trainingData = data.build_full_trainset()
                algo.fit(trainingData)

                prediction = algo.predict(client, thisItem)
                predictedRating = prediction.est

                if 'actual_k' in prediction.details.keys():

                    was_imp = prediction.details['was_impossible']

                    if was_imp: continue

                    actual_k = prediction.details['actual_k']

                    if not ((actual_k < min_k) or (actual_k > max_k)):
                        squaredErr = (predictedRating - actualRating) ** 2
                        squaredErrorList.append(squaredErr)
                else:
                    # print('No actual k')
                    squaredErr = (predictedRating - actualRating) ** 2
                    squaredErrorList.append(squaredErr)
    
    return np.sqrt(np.mean(squaredErrorList)), len(squaredErrorList), pointsTested


def evaluateRMSE():
    folderPath = 'tables_72_DE'

    onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
    onlyfiles.sort()
    onlyfiles.remove('summary.txt')

    # Comment this line out to iterate through all 72 tables.
    # onlyfiles = ['50_Female_30s_Married_> 50K.csv', '47_Female_30s_Single_> 50K.csv']

    mainRMSE = []
    mainAlgo = []
    mainSim = []
    forDf = []
    forDfByTable = []

    totalCount = len(ALGO_LIST) * len(SIM_LIST) * len(onlyfiles)
    counter = 0

    for algoNum in range(len(ALGO_LIST)):

        if ALGO_LIST[algoNum]['name'] == 'KNN_Means':
            kRange = K_RANGE
        else:
            kRange=[0]

        for thisK in kRange:
            for simNum in range(len(SIM_LIST)):

                squaredErrorsByTable = 0
                pointsTestedByTable = 0

                rmseByTable = []
                stdDevByTable = []

                for eachFile in onlyfiles:

                    testDfList, trainDfList, overallDf, success = processData(folderPath + '//' + eachFile)

                    if not success:
                        print('********* NOT ENOUGH DATA *******')
                        print(eachFile)
                        print()
                        continue

                    n = len(testDfList)

                    rmseList = []
                    totalSquaredErrorsForThisTable = 0
                    cumulativePointsTestedForThisTable = 0

                    for i in range(n):
                        print('Algo:', ALGO_LIST[algoNum]['name'], 'Sim Method:', SIM_LIST[simNum]['name'],
                              'File:', eachFile, 'k:', thisK, 'Fold #:', i)
                        print()

                        min_k = thisK
                        max_k = thisK
                        rmse, numSquaredErrors, numPointsTested = measureAccuracy(testDfList[i], trainDfList[i], overallDf, ALGO_LIST[algoNum]['algo'],
                                               SIM_LIST[simNum],
                                               min_k=min_k, max_k=max_k)
                        

                        rmseList.append(rmse)
                        totalSquaredErrorsForThisTable += numSquaredErrors
                        cumulativePointsTestedForThisTable += numPointsTested

                    for i in range(n):
                        print()
                        print('Fold Num:', i, 'Algo:', ALGO_LIST[algoNum]['name'], 'RMSE: %.4f' % (rmseList[i]))

                    overallRMSE = np.mean(rmseList)
                    rmseStdDev = np.std(rmseList)

                    print('****************************************')
                    print('FILE:', eachFile)
                    print('Algo:', ALGO_LIST[algoNum]['name'], 'Sim:', SIM_LIST[simNum]['name'], "k:", thisK)
                    print('Overall Average RMSE:', overallRMSE)
                    print('Overall RMSE Std Dev', rmseStdDev)
                    print('****************************************')

                    rmseByTable.append(overallRMSE)
                    stdDevByTable.append(rmseStdDev)
                    squaredErrorsByTable += totalSquaredErrorsForThisTable
                    pointsTestedByTable += cumulativePointsTestedForThisTable

                    counter += 1

                    print('%.2f percent complete.' % (100 * counter / totalCount))

                    forDfByTable.append([ALGO_LIST[algoNum]['name'], SIM_LIST[simNum]['name'],
                                          eachFile, overallRMSE, rmseStdDev,  min_k, max_k,
                                         totalSquaredErrorsForThisTable, cumulativePointsTestedForThisTable])

                mainRMSE.append(np.mean(rmseByTable))
                mainAlgo.append(ALGO_LIST[algoNum]['name'])
                mainSim.append(SIM_LIST[simNum]['name'])
                forDf.append([mainAlgo[-1], mainSim[-1], mainRMSE[-1], min_k, max_k,
                             squaredErrorsByTable, pointsTestedByTable])

    print(mainAlgo)
    print(mainSim)
    print(mainRMSE)

    mainDf = pd.DataFrame(data=forDf, columns=['Algo', 'Sim_Method', 'RMSE', 'MIN_K', 'MAX_K',
                                               'Total RMSE Points', 'Total Tested Points'])
    tableDf = pd.DataFrame(data=forDfByTable, columns=['Algo', 'Sim_Method','File_Name','RMSE', 'RMSE Std Dev',
                                                       'MIN_K', 'MAX_K','RMSE Points', 'Tested Points'])

    print(mainDf)
    print()
    print(tableDf)

    mainDf.to_csv('results//mainResults_k8.csv')
    tableDf.to_csv('results//tableResults_k8.csv')


def tableSelector(gender, agegroup, lifestage, incomegroup):
    GENDER_GROUP_LIST = ['Male', 'Female']
    LIFESTAGE_GROUP_LIST = ['Single', 'Married', 'Married with dependents']
    AGE_GROUP_LIST = ['20s', '30s', '40s', '50s']
    INCOME_GROUP_LIST = ['<50k', '> 50K', '> 150K']

    filename = ''

    if gender == GENDER_GROUP_LIST[0]:
        if agegroup == AGE_GROUP_LIST[0]:
            if lifestage == LIFESTAGE_GROUP_LIST[0]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '1_Male_20s_Single_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '2_Male_20s_Single_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '3_Male_20s_Single_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[1]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '4_Male_20s_Married_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '5_Male_20s_Married_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '6_Male_20s_Married_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[2]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '7_Male_20s_Married with dependents_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '8_Male_20s_Married with dependents_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '9_Male_20s_Married with dependents_> 150K.csv'

        elif agegroup == AGE_GROUP_LIST[1]:
            if lifestage == LIFESTAGE_GROUP_LIST[0]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '10_Male_30s_Single_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '11_Male_30s_Single_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '12_Male_30s_Single_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[1]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '13_Male_30s_Married_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '14_Male_30s_Married_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '15_Male_30s_Married_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[2]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '16_Male_30s_Married with dependents_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '17_Male_30s_Married with dependents_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '18_Male_30s_Married with dependents_> 150K.csv'

        elif agegroup == AGE_GROUP_LIST[2]:
            if lifestage == LIFESTAGE_GROUP_LIST[0]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '19_Male_40s_Single_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '20_Male_40s_Single_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '21_Male_40s_Single_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[1]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '22_Male_40s_Married_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '23_Male_40s_Married_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '24_Male_40s_Married_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[2]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '25_Male_40s_Married with dependents_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '26_Male_40s_Married with dependents_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '27_Male_40s_Married with dependents_> 150K.csv'

        elif agegroup == AGE_GROUP_LIST[3]:
            if lifestage == LIFESTAGE_GROUP_LIST[0]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '28_Male_50s_Single_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '29_Male_50s_Single_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '30_Male_50s_Single_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[1]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '31_Male_50s_Married_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '32_Male_50s_Married_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '33_Male_50s_Married_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[2]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '34_Male_50s_Married with dependents_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '35_Male_50s_Married with dependents_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '36_Male_50s_Married with dependents_> 150K.csv'

    elif gender == GENDER_GROUP_LIST[1]:
        if agegroup == AGE_GROUP_LIST[0]:
            if lifestage == LIFESTAGE_GROUP_LIST[0]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '37_Female_20s_Single_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '38_Female_20s_Single_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '39_Female_20s_Single_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[1]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '40_Female_20s_Married_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '41_Female_20s_Married_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '42_Female_20s_Married_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[2]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '43_Female_20s_Married with dependents_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '44_Female_20s_Married with dependents_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '45_Female_20s_Married with dependents_> 150K.csv'

        elif agegroup == AGE_GROUP_LIST[1]:
            if lifestage == LIFESTAGE_GROUP_LIST[0]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '46_Female_30s_Single_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '47_Female_30s_Single_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '48_Female_30s_Single_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[1]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '49_Female_30s_Married_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '50_Female_30s_Married_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '51_Female_30s_Married_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[2]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '52_Female_30s_Married with dependents_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '53_Female_30s_Married with dependents_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '54_Female_30s_Married with dependents_> 150K.csv'

        elif agegroup == AGE_GROUP_LIST[2]:
            if lifestage == LIFESTAGE_GROUP_LIST[0]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '55_Female_40s_Single_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '56_Female_40s_Single_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '57_Female_40s_Single_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[1]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '58_Female_40s_Married_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '59_Female_40s_Married_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '60_Female_40s_Married_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[2]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '61_Female_40s_Married with dependents_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '62_Female_40s_Married with dependents_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '63_Female_40s_Married with dependents_> 150K.csv'

        elif agegroup == AGE_GROUP_LIST[3]:
            if lifestage == LIFESTAGE_GROUP_LIST[0]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '64_Female_50s_Single_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '65_Female_50s_Single_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '66_Female_50s_Single_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[1]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '67_Female_50s_Married_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '68_Female_50s_Married_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '69_Female_50s_Married_> 150K.csv'

            elif lifestage == LIFESTAGE_GROUP_LIST[2]:
                if incomegroup == INCOME_GROUP_LIST[0]:
                    filename = '70_Female_50s_Married with dependents_<50k.csv'

                elif incomegroup == INCOME_GROUP_LIST[1]:
                    filename = '71_Female_50s_Married with dependents_> 50K.csv'

                elif incomegroup == INCOME_GROUP_LIST[2]:
                    filename = '72_Female_50s_Married with dependents_> 150K.csv'

    return filename


def generateUserInputs(userID, productIndex, rating):

    entry = {}
    entry['CLIENTS'] = userID
    entry['ITEM'] = PRODUCT_LIST[productIndex]
    entry['RATING'] = rating

    print(entry)

    return entry

def getTopN(data, processedDf, userID, N = 3, algo = ALGO_LIST[1], sim = SIM_LIST[0], kValue = 5):
    print('Getting Top %i Recommendations.'%(N))
    print('Algorithm:', algo['name'])
    print('Similarity Method:', sim['name'])
    print('Client ID:', userID)

    algorithm = algo['algo']
    algorithm.sim_options = sim
    algorithm.min_k = kValue
    algorithm.k = kValue

    products = list(processedDf.ITEM.unique())
    userNonEmpty = list(processedDf[processedDf['CLIENTS'] == userID].ITEM)

    # Don't ask for recommendation on already purchased products

    if len(userNonEmpty) == 0:
        print("ERROR!!\nThis user has not rated any product.")
        exit()

    print('This user has rated %i items.'%(len(userNonEmpty)))

    # for each in userNonEmpty:
    #     print(each)

    trainingSet = data.build_full_trainset()
    algorithm.fit(trainingSet)

    predict_ratings = []

    for el in products:
        if el in userNonEmpty: continue
        prediction = algorithm.predict(userID, el)
        predict_ratings.append([el, prediction.est])

    columns = ['ITEM', 'RATING']

    predict_df = pd.DataFrame(predict_ratings,columns=columns)

    predict_df = predict_df.sort_values(by=[columns[1], columns[0]], ascending=False,)

    allRatings = predict_df.reset_index(drop=True)

    result = predict_df[:N].reset_index(drop=True)

    print(allRatings)

    allNSameRating = True

    moreOutsideN = allRatings.iloc[N-1].RATING == allRatings.iloc[N].RATING

    lastRating = result.iloc[0].RATING

    for i in range(1, len(result)):
        thisRating = result.iloc[i].RATING

        if (lastRating != thisRating):
            allNSameRating = False
            break

    return result, allNSameRating, moreOutsideN


def runGUI():

    prodGUI = []
    for prod in PRODUCT_LIST:
        prodGUI.append([sg.Text(prod), sg.Combo(['NA',1,2], size=(15,1))])


    layout = [
        [sg.Text('Please enter the following information:')],
        [sg.Text('Client ID'), sg.InputText('')],
        [sg.Text('Gender'), sg.Combo(GENDER_GROUP_LIST, size=(15,1)), sg.Text('Age Group'), sg.Combo(AGE_GROUP_LIST, size=(15,1))],
        [sg.Text('Lifestage'), sg.Combo(LIFESTAGE_GROUP_LIST, size=(15,1)), sg.Text('Income Group'), sg.Combo(INCOME_GROUP_LIST, size=(15,1))],
        ]

    layout += prodGUI

    layout += [
        [sg.Submit(), sg.Cancel()]
    ]

    window = sg.Window('Inputs for Top %i Recommendations'%(TOP_N)).Layout([[sg.Column(layout, size=(600,700), scrollable=True)]])
    button, values = window.Read()

    print(button, values)

    return button, values

def getTopNFromGUI():
    while True:
        # run this just to get PRODUCT_LIST populated
        processData('tables_72_DE/50_Female_30s_Married_> 50K.csv')

        button, values = runGUI()

        if button == 'Submit':
            userID = values[0]
            gender = values[1]
            agegroup = values[2]
            lifestage = values[3]
            incomegroup = values[4]

            inputs = values[5:]

        else:
            print('User Cancelled The Action.')
            exit()

        filename = tableSelector(gender = gender, agegroup = agegroup,
                                 lifestage = lifestage, incomegroup= incomegroup)

        print('Reading from file:', filename)

        folderPath = 'tables_72_DE'

        full_file = folderPath + '//' + filename

        df = pd.read_csv(full_file)
        df.drop(labels=['GENDER', 'AGE GROUP', 'LIFESTAGE', 'INCOME GROUP'], inplace=True, axis=1)
        df = processDfInto3Cols(df)

        for i in range(len(inputs)):
            if inputs[i]!='NA':
                entry = generateUserInputs(userID, productIndex=i, rating = inputs[i])
                inputDf = pd.DataFrame([entry])
                df = pd.concat([df,inputDf]).reset_index(drop=True)

        data = Dataset.load_from_df(df, reader=Reader())

        result, allNSameRating, moreOutsideN = getTopN(data, df, userID, N=TOP_N, algo=ALGO_LIST[1], sim=SIM_LIST[0], kValue=5)

        print(result)
        print(allNSameRating)
        print(moreOutsideN)

        resultString  = 'Here are the top %i recommendations:'%(TOP_N)

        for i in range(len(result)):
            resultString += '\n\n'
            resultString += '%i. '%(i+1)
            resultString += result.iloc[i].ITEM
            print(result.iloc[i].ITEM)

        if allNSameRating:
            resultString += '\n\n\n'
            resultString += 'Note: These recommendations have the same predicted ratings.'

        if moreOutsideN:
            resultString += '\n\n\n'
            resultString += 'Note: There are further products with the same predicted ratings outside of these top %i recommendations.'%(TOP_N)


        sg.Popup(resultString+'\n\n\n\n\n')

def plotKs(inputFile = 'results/overallResults.csv'):

    df = pd.read_csv(inputFile)

    df = df[df['MIN_K']!=0]

    kList = list(df.MIN_K)
    rmseList = list(df.RMSE)

    chartMode = 'lines' # or 'markers

    chart = go.Scatter(x = kList, y = rmseList, mode=chartMode)

    fig = go.Figure([chart])

    algo = df.iloc[0]['Algo']
    sim = df.iloc[0]['Sim_Method'].capitalize() + ' Similarity'

    fig.update_layout(title=algo+ ' using ' + sim+'<br>'+'RMSE vs K')

    plotly.offline.plot(fig, filename='Charts//RMSE_vs_K.html')

def plotBarChart(inputFile = 'results/overallResults.csv'):

    df = pd.read_csv(inputFile)

    pearsonDf = df[df['Sim_Method'] == 'pearson']
    # cosineDf = df[df['Sim_Method'] == 'cosine']

    totalTestedPoints = pearsonDf.iloc[0]['Total Tested Points']

    rmseList = list(pearsonDf.RMSE)
    algoList = []

    for i,r in pearsonDf.iterrows():
        if r['Algo'] == 'KNN_Means':
            algoList.append(r['Algo'] + '_k' +str(r['MIN_K']))
        else:
            algoList.append(r['Algo'])

    maxRMSE = max(rmseList) * 1.1
    minRMSE = min(rmseList) * 0.9

    dataRMSE = [
        go.Bar(name='Pearson', x = algoList, y = rmseList),
        # go.Bar(name='Cosine', x=cosineDf.algo, y=cosineDf.RMSE)
    ]


    layoutRMSE = go.Layout(yaxis = {'range': [minRMSE, maxRMSE]}, title = 'Root Mean Square Error, Num of Folds = %i, Points Tested Per Algo = %i'%(NUM_FOLDS, totalTestedPoints))

    figRMSE = go.Figure(
        data = dataRMSE,
              layout=layoutRMSE)


    plotly.offline.plot(figRMSE, filename='Charts/RMSE_vs_Algo.html')


if __name__ == "__main__":

    start = time.time()

    # evaluateRMSE()

    getTopNFromGUI()

    # plotKs()

    # plotBarChart()

    print()
    print('Time taken:', time.time()-start, 's')

# # visualization
# def plotBarChart(file):
#
#     df = pd.read_csv(file)
#
#     pearsonDf = df[df['Sim_Method'] == 'pearson']
#     cosineDf = df[df['Sim_Method'] == 'cosine']
#
#     rmseList = list(df.RMSE)
#
#     maxRMSE = max(rmseList)
#     minRMSE = min(rmseList)
#
#     dataRMSE = [
#         go.Bar(name='Pearson', x = pearsonDf.algo, y = pearsonDf.RMSE),
#         go.Bar(name='Cosine', x=cosineDf.algo, y=cosineDf.RMSE)
#     ]
#
#
#     layoutRMSE = go.Layout(yaxis = {'range': [minRMSE, maxRMSE]}, title = 'Root Mean Square Error, Num of Folds = %i'%(NUM_FOLDS))
#
#     figRMSE = go.Figure(
#         data = dataRMSE,
#               layout=layoutRMSE)
#
#
#     plotly.offline.plot(figRMSE, filename='RMSE.html')
#


# def makeRecommendation(file, algorithm, sim, min_k, max_k, userInputDict, userID = '9999'):
#
#     folderPath = 'tables_72_DE'
#     testDfList, trainDfList, overallDf, success = processData(folderPath + '//' + eachFile, folds=1)
#
#     print(len(testDfList, len(trainDfList)))
#
#     df = testDfList[0]
#
#     newList = []
#
#     for eachLabel in userInputDict.keys():
#         newList.append([userID, eachLabel, userInputDict[eachLabel]])
#
#     newDf = pd.DataFrame(newList, columns=['CLIENTS', 'ITEM', 'RATING'])
#
#     df = pd.concat([df, newDf])
#
#     reader = Reader()
#
#
#     algo = algorithm['algo']
#
#     algo.sim_options = sim
#     algo.min_k = min_k
#     algo.k = max_k
#
#     data = Dataset.load_from_df(df, reader)
#
#     ratedItems = list(df.ITEM)
#     predict_ratings = []
#
#
#     for eachItem in ratedItems:
#         if not (eachItem in userInputDict.keys()):
#             prediction = algo.predict(userID, eachItem)
#             predict_ratings.append([eachItem, prediction.est])
#
#     columns = ['ITEM', 'RATING']
#
#     predict_df = pd.DataFrame(predict_ratings,columns=columns)
#
#     predict_df = predict_df.sort_values(by=columns[1], ascending=False,)
#
#     return predict_df[:N].reset_index(drop=True)












