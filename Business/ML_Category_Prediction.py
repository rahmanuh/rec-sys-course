from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
import csv

def train(data):
    count_vect = TfidfVectorizer(stop_words = 'english',lowercase=True, analyzer ='word')

    train_data = []
    label_data = []
    for id in data.keys():
        text = data.get(id)[1]
        train_data.append(text)
        label_data.append(data.get(id)[0])


    train_vectors = count_vect.fit_transform(train_data)
    #train_vectors.shape
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train_vectors)
    #train_tfidf.shape


    #model = MultinomialNB()
    model = ComplementNB(alpha=0.5)

    model.fit(train_tfidf, label_data)

    return model, tfidf_transformer, count_vect


def predict(textualTest,model,tfidf_transformer,count_vect):
    out_dict = {}
    X_new_counts = count_vect.transform([textualTest])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    model.predict(X_new_tfidf)
    #model.fit(train_tfidf, label_data).predict(X_new_tfidf)

    for prob in model.predict_proba(X_new_tfidf):
        for cat, p in zip(model.classes_, prob):
            #print(cat+":"+str(p))
            out_dict.update({cat: str(p)})
            ranked_dict = {k: v for k, v in sorted(out_dict.items(), key=lambda x: x[1])}

    print(list(ranked_dict.items())[-10::])
    #return ranked_dict

def loadData(dataFile):
    result_data = {}
    with open(dataFile, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for line in reader:
            result_data[line[0]] = [line[1], line[5]]
    return result_data

data = loadData("../DataVideoGame/all_games_PC.csv")

#test = "When a young street hustler, a retired bank robber and a terrifying psychopath find themselves entangled with some of the most frightening and deranged elements of the criminal underworld, the U.S. government and the entertainment industry, they must pull off a series of dangerous heists to survive in a ruthless city in which they can trust nobody, least of all each other."
test = "With yet another edition of the annual NBA Live Basketball series surfacing the question is whether the change is worht the price tag NBA Live 2001 embraces all of the action of the NBA while featuring all of your favorite NBA stars and rookies."
#test = "Arthur Morgan and the Van der Linde gang are outlaws on the run. With federal agents and the best bounty hunters in the nation massing on their heels, the gang must rob, steal and fight their way across the rugged heartland of America in order to survive. As deepening internal divisions threaten to tear the gang apart, Arthur must make a choice between his own ideals and loyalty to the gang who raised him."


model, tfidf_transformer, count_vect = train(data)
predict(test, model, tfidf_transformer, count_vect) # I do not mapped yet, please check the categories manually