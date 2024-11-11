from sklearn.ensemble import RandomForestRegressor

def rf(X_train, X_test, y_train, max_depth, n_estimators, max_features, min_samples_leaf, random_state):
    if (max_depth == None or n_estimators == None):
        classifier = RandomForestRegressor(random_state=random_state)
    else:
        classifier = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state, max_features=max_features, min_samples_leaf=min_samples_leaf)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred
