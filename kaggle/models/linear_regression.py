from sklearn.linear_model import LinearRegression

def create_linear_regression_model(X_train, y_train):
    """Creates and trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
