import pandas as pd
# import plotly.graph_objs as go
# import plotly
from surprise import Reader, Dataset, KNNWithMeans, SlopeOne,SVD, \
    NMF, NormalPredictor, BaselineOnly, CoClustering, KNNBaseline, KNNBasic, KNNWithZScore,SVDpp
import surprise.model_selection as models
import random, time
import numpy as np
from os import listdir
from os.path import isfile, join

# Parameters

SIM_LIST = [
            {"name": "pearson", "user_based": True, "min_support": 1},
            # {"name": "msd", "user_based": True, "min_support": 1},
            # {"name": "cosine", "user_based": True, "min_support": 1}
            ]


ALGO_LIST = [
            {"name": 'SlopeOne', "algo": SlopeOne()},
            {"name": 'KNN_Means', "algo": KNNWithMeans(verbose=False)},
            {"name": 'SVD', "algo": SVD(verbose=False)},
            ]

INCLUDE_NAS_IN_RMSE = False
RNG_SEED = 1
NUM_FOLDS = 5 # must be at least 2
K_RANGE = range(1,8)

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
    print('Gender:',gender, '\nAge Group:',age_group, "\nLifestage:",lifestage, '\nIncome:', incomeLvl, '\n')

    return testDfList, trainDfList, df, True

def processDfInto3Cols(df):
    products = df.columns[1:]  # remove 'CLIENTS'

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

    mainDf.to_csv('results//mainResults.csv')
    tableDf.to_csv('results//tableResults.csv')



def getTopN(df, userID, N = 1, algo = ALGO_LIST[0], sim = SIM_LIST[0]):
    print('Getting Top %i Recommendations.'%(N))
    print('Algorithm:', algo['name'])
    print('Similarity Method:', sim['name'])
    print('Client ID:', userID)

    algorithm = algo['algo']
    algorithm.sim_options = sim

    products = df.columns[1:]
    userEntry = df[df['CLIENTS']== userID]

    # Don't ask for recommendation on already purchased products
    userEmptyItems = []

    for prod in products:

        if userEntry[prod].values[0] == 0:
            userEmptyItems.append(prod)

    if len(userEmptyItems) == 48:
        print("ERROR!!\nThis user has not rated any product.")
        exit()

    print('This user has rated %i items.'%(48 - len(userEmptyItems)))
    data = Dataset.load_from_df(df, reader=Reader())
    trainingSet = data.build_full_trainset()
    algorithm.fit(trainingSet)

    predict_ratings = []

    for el in userEmptyItems:
        prediction = algorithm.predict(userID, el)
        predict_ratings.append([el, prediction.est])

    columns = ['ITEM', 'RATING']

    predict_df = pd.DataFrame(predict_ratings,columns=columns)

    predict_df = predict_df.sort_values(by=columns[1], ascending=False,)

    print(predict_df)

    return predict_df[:N].reset_index(drop=True)

if __name__ == "__main__":

    start = time.time()

    evaluateRMSE()

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
# def evaluateKs(algorithm, sim, file, kRange = [MIN_K,MAX_K],):
#     algo = algorithm['algo']
#
#     algo.sim_options = sim
#     folderPath = 'tables_72_DE'
#
#     kList = []
#     rmseList = []
#
#     for thisK in kRange:
#         algo.min_k = thisK
#         algo.k = thisK
#
#         testDfList, trainDfList, overallDf, success = processData(folderPath + '//' + eachFile)
#
#         if not success:
#             print('********* NOT ENOUGH DATA *******')
#             print(eachFile)
#             print()
#             continue
#
#         n = len(testDfList)
#
#         rmseList = []
#
#         for i in range(n):
#             print('Algo:', ALGO_LIST[algoNum]['name'], 'Sim Method:', SIM_LIST[simNum]['name'],
#                   'File:', eachFile, 'Fold #:', i)
#             print()
#
#             min_k = MIN_K
#             max_k = MAX_K
#             rmse = measureAccuracy(testDfList[i], trainDfList[i], overallDf, ALGO_LIST[algoNum]['algo'],
#                                    SIM_LIST[simNum],
#                                    min_k=min_k, max_k=max_k)
#
#             rmseList.append(rmse)
#
#         for i in range(n):
#             print()
#             print('Fold Num:', i, 'Algo:', ALGO_LIST[algoNum]['name'], 'RMSE: %.4f' % (rmseList[i]))
#
#         overallRMSE = np.mean(rmseList)
#         rmseStdDev = np.std(rmseList)
#
#         print('****************************************')
#         print('FILE:', eachFile)
#         print('Algo:', ALGO_LIST[algoNum]['name'], 'Sim:', SIM_LIST[simNum]['name'])
#         print('k:', thisK)
#         print('Overall Average RMSE:', overallRMSE)
#         print('Overall RMSE Std Dev', rmseStdDev)
#         print('****************************************')
#
#         rmseList.append(overallRMSE)
#         kList.append(thisK)
#
#     chart = go.Scatter(x = kList, y = rmseList, mode='markers')
#
#     fig = go.Figure([chart])
#
#     fig.update_layout(title=algorithm['name'] + ' ' + sim['name'])
#
#     plotly.offline.plot(fig, filename='RMSE_vs_K.html')

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












