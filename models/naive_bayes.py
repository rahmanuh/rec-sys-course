from sklearn.naive_bayes import GaussianNB

def nb(x_train, x_test, y_train):
    naive_bayes = GaussianNB()
    naive_bayes.fit(x_train, y_train)
    y_pred = naive_bayes.predict(x_test)
    return y_pred