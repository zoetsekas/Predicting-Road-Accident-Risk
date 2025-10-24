from sklearn.naive_bayes import GaussianNB

def create_naive_bayes_model(config, X_train, y_train):
    """Creates and trains a Gaussian Naive Bayes model."""
    model = GaussianNB(**config)
    model.fit(X_train, y_train)
    return model
