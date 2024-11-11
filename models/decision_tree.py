from sklearn.tree import DecisionTreeRegressor

def DT(X_train, X_test, y_train, criterion, max_depth, max_features, min_samples_leaf, random_state):
    if (max_depth == None or max_features == None):
        classifier = DecisionTreeRegressor(random_state=random_state)
    else:
        classifier = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, random_state=random_state)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred, classifier