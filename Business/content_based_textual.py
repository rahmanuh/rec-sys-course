import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

def recommend_by_text(dataset, target):
    df = pd.read_csv(dataset)
    #df["summary"].fillna("empty")
    #print(df["summary"])

    #first_release_date,genres,id,name,platforms,summary,storyline,rating,main,extra,completionist,review_score,review_count,people_polled

    #df.loc[len(df.index)] = target we do not add anymore the target in the training

    vd = TfidfVectorizer(stop_words='english')
    x = list(vd.fit_transform(df['summary'].values.astype("U")).toarray())
    #vt = TfidfVectorizer(vocabulary=vd.vocabulary_)

    vectorized_target = vd.transform([target[5]]).toarray()
    #vectorized_target = vd.fit_transform(vectorized_target)

    results = {}
    id_row = 2
    #print(len(x))
    #target = x[len(x)-1]
    #print(target)

    for row in x:
        similarity = cosine_similarity(vectorized_target, row)
        results[id_row] = similarity
        id_row += 1

    ordered_results = {k: v for k, v in sorted(results.items(), key=lambda x: x[1])}
    return list(ordered_results.keys())[-20::]
def cosine_similarity(list1,list2):
    cosine = np.dot(list1, list2) / (norm(list1) * norm(list2))
    #print("Cosine Similarity:", cosine)
    return cosine

def load_slow(dataset, genres_data, target):

    genres = {}
    with open(genres_data, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for line in reader:
            genres[line[0]] = line[1]
    print(genres)

    with open(dataset, newline='') as csvfile:
        reader2 = csv.reader(csvfile, delimiter=',')
        next(reader2)
        matrix = []


        for line in reader2:
            row = []
            line_genres = line[1].replace("[","").replace("]","").replace("\'","").replace(" ","").split(",")
            for genre in genres:
                if (genre in line_genres):
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row)

        target_vector = []
        for genre in genres:
            if (genre in target[1].replace("[","").replace("]","").replace("\'","").split(",")):
                target_vector.append(1)
            else:
                target_vector.append(0)

    print("done")
    return matrix, target_vector
            #row.append(line[1])


def recommend_by_genre(target, games_data, genres_data):

    data, target_vector = load_slow(games_data,genres_data,target)

    results = {}
    count_id = 1
    for game in data:
        similarity = cosine_similarity(target_vector, game)
        count_id += 1
        results[count_id] = similarity
    ordered_results = {k: v for k, v in sorted(results.items(), key=lambda x: x[1])}

    topN = []
    for key in ordered_results:
        if(ordered_results[key]>0.9):
            topN.append(key)

    return topN[-10::] #list(ordered_results.items())[-10::]

def map_names(results, dataset):

    with open(dataset, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        id = 2
        for row in reader:
            if(id in results):
                print(row[3])
            id+=1

def evaluate(gt, results):
    tp = 0
    fp = 0
    fn = len(gt)
    for res in results:
        if(str(res) in gt):
            tp+=1
        else:
            fp+=1
    print("tp: "+str(tp))
    print("fp: "+str(fp))
    print("fn: "+str(fn-tp))

    print("precision: "+str(tp/fp))
    print("recall: "+str(tp/(tp+fn)))

test_text = ('''You've inherited your grandfather's old farm plot in Stardew Valley. Armed with hand-me-down tools and a few coins, you set out to begin your new life. Can you learn to live off the land and turn these overgrown fields into a thriving home? It won't be easy. Ever since Joja Corporation came to town, the old ways of life have all but disappeared. The community center, once the town's most vibrant hub of activity, now lies in shambles. But the valley seems full of opportunity. With a little dedication, you might just be the one to restore Stardew Valley to greatness!

Features

    Turn your overgrown field into a lively farm! Raise animals, grow crops, start an orchard, craft useful machines, and more! You'll have plenty of space to create the farm of your dreams.

    8 Player Farming! Invite 1-7 players to join you in the valley online! Players can work together to build a thriving farm, share resources, and improve the local community. As more hands are better than one, players have the option to scale profit margin on produce sold for a more challenging experience.

    Improve your skills over time. As you make your way from a struggling greenhorn to a master farmer, you'll level up in 5 different areas: farming, mining, combat, fishing, and foraging. As you progress, you'll learn new cooking and crafting recipes, unlock new areas to explore, and customize your skills by choosing from a variety of professions.

    Become part of the local community. With over 30 unique characters living in Stardew Valley, you won't have a problem finding new friends! Each person has their own daily schedule, birthday, unique mini-cutscenes, and new things to say throughout the week and year. As you make friends with them, they will open up to you, ask you for help with their personal troubles, or tell you their secrets! Take part in seasonal festivals such as the luau, haunted maze, and feast of the winter star.

    Explore a vast, mysterious cave. As you travel deeper underground, you'll encounter new and dangerous monsters, powerful weapons, new environments, valuable gemstones, raw materials for crafting and upgrading tools, and mysteries to be uncovered.

    Breathe new life into the valley. Since JojaMart opened, the old way of life in Stardew Valley has changed. Much of the town's infrastructure has fallen into disrepair. Help restore Stardew Valley to it's former glory by repairing the old community center, or take the alternate route and join forces with Joja Corporation.

    Court and marry a partner to share your life on the farm with. There are 12 available bachelors and bachelorettes to woo, each with unique character progression cutscenes. Once married, your partner will live on the farm with you. Who knows, maybe you'll have kids and start a family?

    Spend a relaxing afternoon at one of the local fishing spots. The waters are teeming with seasonal varieties of delicious fish. Craft bait, bobbers, and crab pots to help you in your journey toward catching every fish and becoming a local legend!

    Donate artifacts and minerals to the local museum.

    Cook delicious meals and craft useful items to help you out. With over 100 cooking and crafting recipes, you'll have a wide variety of items to create. Some dishes you cook will even give you temporary boosts to skills, running speed, or combat prowess. Craft useful objects like scarecrows, oil makers, furnaces, or even the rare and expensive crystalarium.''')
target = ['123456', "[12,13,15,32]", '123456', 'Stardew Valley', 'nan', test_text, 'Missing','nan','nan','nan','nan','nan','nan','nan']

#results = recommend_by_genre(target, "../DataVideoGame/all_games_PC.csv", "../DataVideoGame/genres.csv")

results = recommend_by_text("../DataVideoGame/all_games_PC.csv", target)

map_names(results,"../DataVideoGame/all_games_PC.csv")

gt = ["19339","40703","48726","61788","3036","41533","16964","32595","42776","42777","49666"]

evaluate(gt,results)