from sklearn.neighbors import KNeighborsRegressor

def nn(xtrain, xtest, y_train, n_neighbor):
    if n_neighbor == None:
        neigh = KNeighborsRegressor()
    else:
        neigh = KNeighborsRegressor(n_neighbors=n_neighbor)
    neigh.fit(xtrain, y_train)
    y_pred = neigh.predict(xtest)
    return y_pred