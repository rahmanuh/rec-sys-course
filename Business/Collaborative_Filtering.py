import random
import time

import numpy
import pandas as pd
import nltk
import csv
import numpy as np
from numpy import sort
from numpy import mean
from numpy.linalg import norm
from scipy import stats

# TODO: Look into movies.csv and give high rating ( >= 4.0 ) on action and thriller movies
def rate_my_movies():
    user_ids = []
    movie_ids = []
    ratings = []
    timestamps = []

    for i in range(1500):
        user_ids.append(611)
        movie_ids.append(random.randint(1,193609))
        ratings.append(round(random.uniform(1.0, 5.0) * 2) / 2)
        timestamps.append(int(time.time()) + i)

    my_movie_data = {
        'userId': user_ids,
        'movieId': movie_ids,
        'rating': ratings,
        'timestamp': timestamps
    }

    return my_movie_data

# This is just a utility method which will be called only ONCE separately
def store_my_movie_data():
    data = rate_my_movies()
    df = pd.DataFrame(data)
    df.sort_values(by=['movieId'], inplace=True)
    df.to_csv("../Data/my_movie_data.csv", sep=',', encoding='utf-8', index=False)

###### Loading Dataset ######
def createUserRatings(dataset):

    df = pd.read_csv(dataset)

    # Add my movie ratings
    my_df = pd.read_csv("../Data/my_movie_data.csv")
    df = pd.concat([df, my_df], ignore_index=True)

    movie_ids = df['movieId'].explode().unique()
    user_ids = df['userId'].explode().unique()
    movie_ids.sort()

    matrix = []
    #with open(dataset, newline='') as csvfile:
    #    reader = csv.reader(csvfile, delimiter=',')
    #    next(reader)

    #creating a matrix [5.0,0,1.0] each row show the rating of a user for all the movies
    for user in user_ids:
        row = []
        userDF = (df[df['userId'] == user])
        rated_movies = userDF["movieId"].values
        for movie in movie_ids:
            if(movie in rated_movies):
                row.append(userDF.loc[userDF['movieId'] == movie, "rating"].iloc[0])
            else:
                row.append(0)
        matrix.append(row)
    return matrix

###### Similarity Metrics ######

def cosine_similarity(list1,list2):
    cosine = np.dot(list1, list2) / (norm(list1) * norm(list2))
    print("Cosine Similarity:", cosine)
    return cosine
def cosine_similarity_fixed(list1,list2):
    short_l1 = []
    short_l2 = []
    for l1, l2 in zip(list1, list2):
        if(l1 != 0 and l2 != 0):
            short_l1.append(l1)
            short_l2.append(l2)
    if(len(short_l2)>2):
        cosine = np.dot(short_l1, short_l2) / (norm(short_l1) * norm(short_l2))
    else:
        cosine = 0.0

    return cosine
def pearsons_correlation(list1,list2):
    short_l1 = []
    short_l2 = []
    for l1, l2 in zip(list1, list2):
        if(l1 != 0 and l2 != 0):
            short_l1.append(l1)
            short_l2.append(l2)
    mean_rating = mean(short_l2)
    return(stats.pearsonr(short_l1,short_l2)[0],mean_rating)

###### Functions For User Based ######

def select_users(movieId, data):
    # I want to filter the users who voted the target movie
    short_matrix = []
    for ratings in data:
        if(ratings[int(movieId)]!=0):
            short_matrix.append(ratings)
    return short_matrix
def user_based_recommendation(target,data,movie):

    #print(len(data))
    top_users = {}
    for user in data:
        #print(cosine_similarity_fixed(target, user))
        similarity, mean_rating = pearsons_correlation(target, user)
        if(not numpy.isnan(similarity)):
            rating = user[int(movie)]
            top_users[random.randint(1,1000)] = [similarity, rating, mean_rating]

    ordered_results = {k: v for k, v in sorted(top_users.items(), key=lambda x: x[1][0])}

    topk = list(ordered_results.items())[-10::]
    mean_rating_target = mean(list(filter((0).__ne__, target)))
    den = 0
    num = 0
    for t in topk:
        print("top10", t)
        rating = t[1][1]
        mean_rating = t[1][2]
        similarity = t[1][0]
        num = num + (similarity*(rating-mean_rating))
        den = den+similarity

    #print(num)
    #print(den)
    print(f'Suggested rating: {mean_rating_target+(num/den)}')

###### Functions For Item Based ######

def modify_data_for_item_recommendation(data):

    for user in data:
        mean_rating = mean(list(filter((0).__ne__, user)))
        for i in range(0, len(user)):
            if(user[i]!=0):
                user[i] = user[i] - mean_rating
    return data
def item_based_recommendation(user, data, movieId):
    print(user)
    trasposed_matrix = np.array(data).transpose()
    target = trasposed_matrix[int(movieId)]
    #print(target)
    top_movies = {}
    count = 0
    for movie in trasposed_matrix:
        if(user[count]!=0):
            similarity = cosine_similarity_fixed(target, movie)
            if (not numpy.isnan(similarity)):
                top_movies[random.randint(1, 10000)] = [similarity, count]
        count += 1

    ordered_results = {k: v for k, v in sorted(top_movies.items(), key=lambda x: x[1])}

    topk = list(ordered_results.items())[-10::]

    den = 0
    num = 0
    for t in topk:
        print("top10: ", t)
        target_rating = user[t[1][1]]
        movie_similarity = t[1][0]
        num = num +  target_rating*movie_similarity
        den = den + movie_similarity

    # print(num)
    # print(den)
    print(f"Suggested Rating: {num/den}")


###### Common Procedures ######

my_user_id_idx = 610
# matrix = createUserRatings("../Data/ratings.csv")
matrix = createUserRatings("../Data/ratings.csv")
short_matrix = select_users('8876', matrix)
user_votes = list(matrix[my_user_id_idx])

###### Get User Based Recommendation ######
print(user_votes)
user_based_recommendation(user_votes, short_matrix, '8876')

###### Get Item Based Recommendation ######
data_for_item_rec = modify_data_for_item_recommendation(short_matrix)
item_based_recommendation(user_votes, data_for_item_rec, '8876')