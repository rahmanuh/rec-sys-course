import pandas as pd
import csv

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from models import nearest_neighbors, randomForest, naive_bayes


from sklearn.feature_extraction.text import TfidfVectorizer


def loadCSV(dataset):

    df = pd.read_csv(dataset, nrows=5000) #can take a long if more than 10000
    vd = TfidfVectorizer(stop_words='english')
    df['review_score'].fillna(value=0, inplace=True) #WARNING - put 0 to fill missing values
    df = df[df.review_score != 0]

    df['summary'].fillna(value="Empty", inplace=True)
    #print(df.index[df['name']=='Stardew Valley'])

    Y = df[['review_score','id']]
    #x = list(vd.fit_transform(df['summary'].values.astype("U")).toarray())
    X_text_tfidf = vd.fit_transform(df['summary'])

    # Combine numerical and textual features
    X_combined = pd.DataFrame(X_text_tfidf.toarray())
    #X_combined = pd.concat([df['genres'], pd.DataFrame(X_text_tfidf.toarray())], axis=1)
    X_combined = X_combined.rename(str, axis="columns")
    #print(X_combined.head())

    # train --- genres - vector(tfidf) -> rating
    # test --- genres - vector(tfidf) -> predict

    return X_combined, Y['review_score']

def vectorize_target(target):
    '''
1456444800,"[12, 13, 15, 32]",17000,Stardew Valley,"[3, 6, 14, 34, 39, 41, 46, 48, 49, 130]",
Stardew Valley is an openended countrylife RPG Youve inherited your grandfathers old farm plot in Stardew Valley Armed with handmedown tools and a few coins you set out to begin your new life Can you learn to live off the land and turn these overgrown fields into a thriving home It wont be easy Ever since Joja Corporation came to town the old ways of life have all but disappeared The community center once the towns most vibrant hub of activity now lies in shambles But the valley seems full of opportunity With a little dedication you might just be the one to restore Stardew Valley to greatness,Youve inherited your grandfathers old farm plot in Stardew Valley Armed with handmedown tools and a few coins you set out to begin your new life Can you learn to live off the land and turn these overgrown fields into a thriving home It wont be easy Ever since Joja Corporation came to town the old ways of life have all but disappeared The community center once the towns most vibrant hub of activity now lies in shambles But the valley seems full of opportunity With a little dedication you might just be the one to restore Stardew Valley to greatness,
86.72069532506818,52.58,90.06,159.36,88.0,1760.0,1309.0

    '''
    right_prediction = 86
    vd = TfidfVectorizer(stop_words='english')
    vectorized_target = vd.transform("Stardew Valley is an openended countrylife RPG Youve inherited your grandfathers old farm plot in Stardew Valley Armed with handmedown tools and a few coins you set out to begin your new life Can you learn to live off the land and turn these overgrown fields into a thriving home It wont be easy Ever since Joja Corporation came to town the old ways of life have all but disappeared The community center once the towns most vibrant hub of activity now lies in shambles But the valley seems full of opportunity With a little dedication you might just be the one to restore Stardew Valley to greatness,Youve inherited your grandfathers old farm plot in Stardew Valley Armed with handmedown tools and a few coins you set out to begin your new life Can you learn to live off the land and turn these overgrown fields into a thriving home It wont be easy Ever since Joja Corporation came to town the old ways of life have all but disappeared The community center once the towns most vibrant hub of activity now lies in shambles But the valley seems full of opportunity With a little dedication you might just be the one to restore Stardew Valley to greatness").toarray()
    vectorized_target


def predict_rating(X_train, X_test, y_train):
    max_depth = None
    max_features = None
    min_sample_leaf = None
    n_estimators = None
    random_state = 42
    n_neighbors = 3
    #predictions = randomForest.rf(X_train, X_test, y_train, max_depth, n_estimators, max_features, min_sample_leaf, random_state)

    #predictions = decisionTree.DT(X_train, X_test, y_train, criterion=None, max_depth=None, max_features=None, min_samples_leaf=None, random_state=42)
    #predictions = nearest_neighbors.nn(X_train, X_test, y_train, n_neighbors)
    predictions = naive_bayes.nb(X_train, X_test, y_train)
    return predictions

def tts_version(X_Data, Y_Data):
    X_train, X_test, y_train, y_test = train_test_split(X_Data, Y_Data, stratify=None, random_state=42,
                                                        test_size=0.1, shuffle=True)

    return X_train, X_test, y_train, y_test

X_Data, Y_Data = loadCSV("../DataVideoGame/all_games_PC.csv")

X_train, X_test, y_train, y_test = tts_version(X_Data, Y_Data)

y_predictions = predict_rating(X_train, X_test, y_train)

print("y_predictions:\n", y_predictions)
print("y_test:\n", y_test)

