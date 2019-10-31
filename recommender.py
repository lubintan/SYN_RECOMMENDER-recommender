import pandas as pd
import plotly.graph_objs as go
import plotly
from surprise import Reader, Dataset, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, SlopeOne,SVD,SVDpp, \
    NMF, NormalPredictor, BaselineOnly, CoClustering
import surprise.model_selection as models
import random, time
import numpy as np


# Parameters

RATING_SCALE = (0,2)
SIM_LIST = [
            {"name": "pearson", "user_based": True, "min_support": 1},
            # {"name": "msd", "user_based": True, "min_support": 1},
            # {"name": "cosine", "user_based": True, "min_support": 1}
            ]
RNG_SEED = 22
NUM_FOLDS = 5

ALGO_LIST = [
            {"name": 'KNN_Baseline', "algo": KNNBaseline()},
            {"name": 'KNN_Basic', "algo": KNNBasic()},
            {"name": 'KNN_Means', "algo": KNNWithMeans()},
            {"name": 'KNN_ZScore', "algo": KNNWithZScore()},
            {"name": 'SlopeOne', "algo": SlopeOne()},
            {"name": 'SVD', "algo": SVD()},
            {"name": 'BaselineOnly', "algo": BaselineOnly()},
            ]

PREDICTION_LIST = [KNNBaseline(), KNNBasic(), KNNWithMeans(),KNNWithZScore(), SlopeOne()]
X = ['KNN_Baseline', 'KNN_Basic', 'KNN_Means', 'KNN_ZScore', 'SlopeOne']
FOLD_METHOD = [models.KFold(n_splits=NUM_FOLDS,random_state=RNG_SEED)]
ACCURACY_LIST = ['RMSE', 'MAE']


# For Actual Use

def processData(inputFile = 'tables_72_DE/46_Female_30s_Single_<50k', folds = 5):


    print('Reading data..')
    df = pd.read_csv(inputFile)

    df2 = pd.read_csv('tables_72/46_F_30s_Single_Below 4000')

    print(len(df))
    print(len(df2))

    df = pd.concat([df,df2], sort=False)


    # print(len(df))
    #
    # print(df)

    # df = pd.read_excel(inputFile)

    gender = df.iloc[0]['GENDER']
    age_group = df.iloc[0]['AGE GROUP']
    lifestage = df.iloc[0]['LIFESTAGE']
    incomeLvl = df.iloc[0]['INCOME GROUP']

    df.drop(labels=['GENDER', 'AGE GROUP', 'LIFESTAGE', 'INCOME GROUP','AGE'], inplace=True, axis=1)
    # df.drop(labels=['GENDER', 'AGE GROUP', 'LIFESTAGE', 'INCOME GROUP'], inplace=True, axis=1)

    df.to_csv("outputCombined.csv")


    products = df.columns[1:] # remove 'CLIENTS'

    newList = []

    for i,r in df.iterrows():
        for label in products:
            thisEntry = []
            thisEntry.append(int(r['CLIENTS']))
            thisEntry.append(label)
            thisEntry.append(r[label])
            newList.append(thisEntry)

    processedDf = pd.DataFrame(newList, columns=['CLIENTS', 'ITEM', 'RATING'])

    # drop 0s
    processedDf = processedDf.dropna()

    processedDf = processedDf[processedDf['RATING']!=0].reset_index(drop=True)

    print(processedDf)

    # exit()

    length = len(processedDf)

    if length < folds:
        print("Data too small for folding %i times" % (folds))
        return 0, 0

    else:

        chunkSize = int(length / folds)

        trainDfList = []
        testDfList = []

        for i in range(folds):
            head = i*chunkSize
            tail = (i+1)*chunkSize

            testDf = processedDf[head:tail]
            trainDf = pd.concat([processedDf[0:head], processedDf[tail:]])

            testDfList.append(testDf)
            trainDfList.append(trainDf)


    # reader = Reader(rating_scale=RATING_SCALE)
    # data = Dataset.load_from_df(processedDf, reader)

    print('Data read complete.')
    print()
    print('Gender:',gender, '\nAge Group:',age_group, "\nLifestage:",lifestage, '\nIncome:', incomeLvl, '\n')

    return testDfList, trainDfList

def measureAccuracy(testSet, trainSet, algo=None, sim=SIM_LIST[0]):
    MIN_ENTRIES = 2

    userCount = testSet.groupby('CLIENTS').count()
    userCount = userCount.sort_values(['RATING'], ascending=[0])
    userCount = userCount[userCount['RATING'] >= MIN_ENTRIES]

    userIDs = list(userCount.index)

    itemCount = testSet.groupby('ITEM').count()
    items = list(itemCount.index)

    newTest = []

    # for i,r in testSet.iterrows():
    #     if r.CLIENTS in userIDs:
    #         newTest.append([r.CLIENTS, r.ITEM, r.RATING])
    #
    # filteredTest = pd.DataFrame(newTest, columns=['CLIENTS', 'ITEM', 'RATING'])

    reader = Reader(rating_scale=RATING_SCALE)
    algo.sim_options = sim

    squaredErrorList = []

    for client in userIDs:
        thisDf = testSet[testSet['CLIENTS'] == client].reset_index(drop=True)

        num = len(thisDf)

        for i in range(num):
            actualRating = thisDf.iloc[i].RATING
            thisItem = thisDf.iloc[i].ITEM

            newDf = thisDf.drop(thisDf.index[i])
            trainSet = pd.concat([trainSet, newDf])

            data = Dataset.load_from_df(trainSet, reader)
            trainingData = data.build_full_trainset()
            algo.fit(trainingData)

            # for eachItem in items:
            prediction = algo.predict(client, thisItem)

            squaredErr = (prediction.est - actualRating)**2

            squaredErrorList.append(squaredErr)

    return np.sqrt(np.mean(squaredErrorList))




def getTopN(data, df, userID, N = 1, algo = ALGO_LIST[0], sim = SIM_LIST[0]):
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

    trainingSet = data.build_full_trainset()
    algorithm.fit(trainingSet)

    predict_ratings = []

    for el in userEmptyItems:
        prediction = algorithm.predict(userID, el)
        predict_ratings.append([el, prediction.est])

    columns = ['ITEM', 'RATING']

    predict_df = pd.DataFrame(predict_ratings,columns=columns)

    predict_df = predict_df.sort_values(by=columns[1], ascending=False,)

    return predict_df[:N].reset_index(drop=True)

def compareAlgos(data):
    for i in range(len(ALGO_LIST)):
        algorithm = ALGO_LIST[i]['algo']

        for k in range(len(SIM_LIST)):
            algorithm.sim_options = SIM_LIST[k]

            for n in range(len(FOLD_METHOD)):
                results = models.cross_validate(algorithm, data, measures= ACCURACY_LIST ,
                                                cv=FOLD_METHOD[n], verbose=False)

                rmseMean = np.mean(results['test_rmse'])
                rmseStdDev = np.std(results['test_rmse'])
                maeMean = np.mean(results['test_mae'])
                maeStdDev = np.std(results['test_mae'])

                print()
                print('Algorithm:', ALGO_LIST[i]['name'])
                print('\tSimilarity Method:', SIM_LIST[k]['name'])
                print('\t\tNumber of Folds:', NUM_FOLDS)
                print('\t\tMean RMSE: %4f\tRMSE Std Dev: %.4f'%(rmseMean, rmseStdDev))
                print('\t\tMean MAE: %4f\tMAE Std Dev: %.4f'%(maeMean, maeStdDev))

        print()

if __name__ == "__main__":

    start = time.time()

# The following steps gets the Top N recommendations for a particular user.
# Requires the user to have made at least 1 rating.

    testDfList, trainDfList = processData( folds=4)

    n = len(testDfList)
    overallAccuracy = []

    for algoNum in range(len(ALGO_LIST)):

        for simNum in range(len(SIM_LIST)):

            rmseList = []

            for i in range(n):
                rmse = measureAccuracy(testDfList[i], trainDfList[i], ALGO_LIST[algoNum]['algo'], SIM_LIST[simNum])

                rmseList.append(rmse)

            for i in range(n):
                print()
                print('Fold Num:', i, 'Algo:', ALGO_LIST[algoNum]['name'], 'RMSE: %.4f' % (rmseList[i]))

            overallRMSE = np.mean(rmseList)
            rmseStdDev = np.std(rmseList)

            print('****************************************')
            print('Algo:', ALGO_LIST[algoNum]['name'],'Sim:', SIM_LIST[simNum]['name'])
            print('Overall Average RMSE:', overallRMSE)
            print('Overall RMSE Std Dev', rmseStdDev)
            print('****************************************')

            overallAccuracy.append([ALGO_LIST[algoNum]['name'], SIM_LIST[simNum]['name'], overallRMSE, rmseStdDev])


    print()
    print('FINAL RESULTS')

    for each in overallAccuracy:
        print(each)


    print()
    print('Time taken:', time.time()-start, 's')




############### END ################
# The following code below is for reference for building future functions.
# region: Future Reference

def ratingsDistribution(df):
    data = df['bookRating'].value_counts().sort_index(ascending=False)
    trace = go.Bar(x = data.index,
                   text = ['{:.1f} %'.format(val) for val in (data.values / df.shape[0] * 100)],
                   textposition = 'auto',
                   textfont = dict(color = '#000000'),
                   y = data.values,
                   )
    # Create layout
    layout = dict(title = 'Distribution Of {} book-ratings'.format(df.shape[0]),
                  xaxis = dict(title = 'Rating'),
                  yaxis = dict(title = 'Count'))
    # Create plot
    fig = go.Figure(data=[trace], layout=layout)
    plotly.offline.plot(fig)

def ratingsCountDistribution(df):
    # Number of ratings per book
    data = df.groupby('ISBN')['bookRating'].count().clip(upper=50)

    # Create trace
    trace = go.Histogram(x = data.values,
                         name = 'Ratings',
                         xbins = dict(start = 0,
                                      end = 50,
                                      size = 2))
    # Create layout
    layout = go.Layout(title = 'Distribution Of Number of Ratings Per Book (Clipped at 50)',
                       xaxis = dict(title = 'Number of Ratings Per Book'),
                       yaxis = dict(title = 'Count'),
                       bargap = 0.2)

    # Create plot
    fig = go.Figure(data=[trace], layout=layout)
    plotly.offline.plot(fig)

def ratingsPerUser(df):
    # Number of ratings per user
    data = df.groupby('userID')['bookRating'].count().clip(upper=50)

    # Create trace
    trace = go.Histogram(x = data.values,
                         name = 'Ratings',
                         xbins = dict(start = 0,
                                      end = 50,
                                      size = 2))
    # Create layout
    layout = go.Layout(title = 'Distribution Of Number of Ratings Per User (Clipped at 50)',
                       xaxis = dict(title = 'Ratings Per User'),
                       yaxis = dict(title = 'Count'),
                       bargap = 0.2)

    # Create plot
    fig = go.Figure(data=[trace], layout=layout)
    plotly.offline.plot(fig)

def reduceDimensionality(df):
    min_book_ratings = 50
    filter_books = df['ISBN'].value_counts() > min_book_ratings
    filter_books = filter_books[filter_books].index.tolist()

    min_user_ratings = 50
    filter_users = df['userID'].value_counts() > min_user_ratings
    filter_users = filter_users[filter_users].index.tolist()

    df_new = df[(df['ISBN'].isin(filter_books)) & (df['userID'].isin(filter_users))]
    print('The original data frame shape:\t{}'.format(df.shape))
    print('The new data frame shape:\t{}'.format(df_new.shape))

    return df_new

def readData(df):
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(df[['userID', 'ISBN', 'bookRating']], reader)
    return data

def evaluateAlgos():
    ### test area

    #
    # results = models.cross_validate(KNNWithMeans(sim_options=dict(name='pearson')), data, measures=['rmse'], cv=models.KFold(shuffle=False), verbose=False)
    # results2 = models.cross_validate(KNNBasic(sim_options=dict(name='pearson')), data, measures=['rmse'], cv=models.KFold(shuffle=False), verbose=False)
    #
    # print(results['test_rmse'])
    # print(results2['test_rmse'])
    #
    # exit()
    ### end test area

    user = pd.read_csv('BX_data//BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    user.columns = ['userID', 'Location', 'Age']
    rating = pd.read_csv('BX_data//BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    rating.columns = ['userID', 'ISBN', 'bookRating']
    df = pd.merge(user, rating, on='userID', how='inner')
    df.drop(['Location', 'Age'], axis=1, inplace=True)

    data_length = 3000
    random.seed(RNG_SEED)
    head = random.randint(0, len(df) - 500)
    df = df[head:head + 500]
    print('head index:', head)

    data = readData(df)

    rmseList = []
    maeList = []


    # Iterate over all algorithms
    for foldType in FOLD_METHOD:
        for simType in SIM_LIST:
            thisRmseList = []
            thisMaeList = []

            print()

            for i in range(len(PREDICTION_LIST)):

                algorithm = PREDICTION_LIST[i]

                algorithm.sim_options = simType

                # print(algorithm.sim_options)

                results = models.cross_validate(algorithm, data, measures= ACCURACY_LIST , cv=foldType, verbose=False)
                # print(results['test_rmse'])
                # print()

                rmseMean = np.mean(results['test_rmse'])
                maeMean = np.mean(results['test_mae'])

                thisRmseList.append(rmseMean)
                thisMaeList.append(maeMean)

            rmseList.append(thisRmseList)
            maeList.append(thisMaeList)

    dataRMSE = [
        go.Bar(name='pearson', x = X, y = rmseList[0]),
        go.Bar(name='msd', x=X, y=rmseList[1]),
        go.Bar(name='cosine', x=X, y=rmseList[2])
    ]

    dataMAE = [
        go.Bar(name='pearson', x=X, y=maeList[0]),
        go.Bar(name='msd', x=X, y=maeList[1]),
        go.Bar(name='cosine', x=X, y=maeList[2])
    ]

    layoutRMSE = go.Layout(yaxis = {'range': [
        min(min(rmseList[0]),min(rmseList[1]),min(rmseList[2])),
                  max(max(rmseList[0]),max(rmseList[1]),max(rmseList[2]))]

    }, title = 'Root Mean Square Error, Num of Folds = %i'%(NUM_FOLDS))

    layoutMAE = go.Layout(yaxis={'range': [
        min(min(maeList[0]), min(maeList[1]), min(maeList[2])),
        max(max(maeList[0]), max(maeList[1]), max(maeList[2]))]

    }, title = 'Mean Average Error, Num of Folds = %i'%(NUM_FOLDS))

    figRMSE = go.Figure(
        data = dataRMSE,
              layout=layoutRMSE)

    figMAE = go.Figure(data=dataMAE, layout = layoutMAE)

    plotly.offline.plot(figRMSE, filename='RMSE.html')
    plotly.offline.plot(figMAE, filename='MAE.html')

    print()
    print('Algorithm\t|SimType\t|Avg RMSE |Avg MAE\t')
    for i in range(len(X)):
        for j in range(len(SIM_LIST)):
            print(X[i], '\t', SIM_LIST[j]['name'], '\t%.12f'%(rmseList[j][i]), '\t%.12f'%(maeList[j][i]))

# endregion












