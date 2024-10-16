import pandas as pd
import nltk
import csv
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

def loadCSV(dataset):

    df = pd.read_csv(dataset)
    genres = df['genres'].str.split('|').explode().unique()

    with open(dataset, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        matrix = []
        for line in reader:
            row = []
            row.append(line[0])
            row.append(line[1])
            for genre in genres:
                if(genre in line[2].split('|')):
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row)
    return matrix

def cosine_similarity(list1,list2):
    cosine = np.dot(list1, list2) / (norm(list1) * norm(list2))
    #print("Cosine Similarity:", cosine)
    return cosine

def recommendCF(target, matrix):
    results = {}

    for movie in matrix:
        movie_vect = np.array(movie[2::])
        cosine = cosine_similarity(np.array(target[2::]), movie_vect)
        results[movie[0]] = [movie[1], cosine]

    ordered_results = {k: v for k, v in sorted(results.items(), key=lambda x: x[1][1])}

    return list(ordered_results.items())[-10::]

def build_profile(idUser,data_Matrix,ratings):

    with open(ratings, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        watched_movies = []
        profile = []
        for line in reader:
            if(line[0]==idUser):
                watched_movies.append(line)

        for movie in watched_movies:
            for row in data_Matrix:
                if(movie[1]==row[0]): #watched movie == movie in the dataset then i put it in the profile
                    profile.append(row+[movie[2]])
                    break

    return profile

def recommendantion_with_profile(target,profile):

    results = {}
    for movie in profile:
        movie_vect = np.array(movie[2:len(movie)-1]) #categories
        cosine = cosine_similarity(np.array(target[2:len(movie)-1]), movie_vect)
        results[movie[0]] = [movie[1], cosine, movie[-1]]

    ordered_results = {k: v for k, v in sorted(results.items(), key=lambda x: x[1][1])}

    topk = list(ordered_results.items())[-10::]
    den = 0
    num = 0
    for t in topk:
        #print(t)
        movie_similarity = t[1][1]
        profile_rating = float(t[1][2])
        num = num + movie_similarity*profile_rating
        den = den + movie_similarity

    #print(num/den)
    return(num/den)



###### Common functions ######
movie_matrix = loadCSV("../Data/movies.csv")

###### One Item ######
def one_item():
    print(movie_matrix[0])
    results = recommendCF(movie_matrix[0], movie_matrix)

    for movie in results:
        print(movie[1][0])
    return results

#one_item()

###### Profile ######
def user_profile():
    profile = build_profile('1', movie_matrix, "../Data/ratings.csv")
    return profile

def evaluate_user_profile():
    #we compare a set of movies with the created profile to predict a possibile rating based on the characteristics of the movie already watched
    profile = user_profile()
    predicted_ratings = []
    actual_ratings = []
    for watched_movie in profile:
        #print(watched_movie)
        predicted_rating = recommendantion_with_profile(watched_movie, profile)
        predicted_ratings.append(predicted_rating)
        actual_ratings.append(float(watched_movie[len(watched_movie)-1]))
    print(actual_ratings)
    print(predicted_ratings)
    print("MAE:"+str(mean_absolute_error(actual_ratings, predicted_ratings)))
    print("RMSE:"+str(root_mean_squared_error(actual_ratings, predicted_ratings)))

def evaluate_one_item(ratings, movie_matrix, target_movie):
    print(target_movie)
    similar_movies = one_item()
    similar_movies_list = []
    for sim in similar_movies:
        similar_movies_list.append(sim[0])

    ##### find user who have rated this movie #####
    with open(ratings, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        ratings_list = list(reader)
        users_watched_movie = {}
        for line in ratings_list:
            if (line[1] == target_movie[0]):
                users_watched_movie[line[0]] = []

        for user in users_watched_movie:
            for line in ratings_list:
                if (user == line[0] and line[1]!=target_movie[0] and float(line[2])>1.0): #removed target movie from the profiles
                    users_watched_movie[user].append(line[1])

        tp = 0
        fp = 0
        for similar_movie in similar_movies_list:
            for user in users_watched_movie:
                if(similar_movie in users_watched_movie[user]):
                    tp += 1
                else:
                    fp += 1
        print("True positive: "+str(tp))
        print("False Positive: "+str(fp))

#evaluate_one_item("../Data/ratings.csv","ciao", movie_matrix[0])
evaluate_user_profile()