# Linear Regression as a Meta-Model

In this project, Linear Regression is used not as a primary model for predicting accident risk from the original features, but as a **meta-model** in a stacking ensemble. Its job is to learn the best way to combine the predictions from the more complex base models (LightGBM, XGBoost, etc.).

## Role in the Ensemble

The inputs to the Linear Regression model are not the raw data features (like `speed_limit` or `weather`). Instead, its inputs are the outputs (the predictions) of the base models. It learns a simple linear formula to combine these predictions to produce the final, blended output.

## Advantages

- **Simplicity and Speed**: As a meta-model, Linear Regression is extremely fast to train. Since the base models have already done the heavy lifting of feature extraction and pattern recognition, the meta-model's job is simply to find the best linear combination of their outputs.

- **Interpretability**: The coefficients of the trained Linear Regression model can provide insights into how the ensemble is weighting the predictions of the base models. A larger coefficient for a particular base model suggests that the ensemble relies more heavily on its predictions.

- **Reduces Overfitting**: Using a simple model like Linear Regression as the meta-learner is a good way to prevent overfitting. A more complex meta-model could start to memorize the noise from the base models' predictions, but Linear Regression is constrained to finding a simple, robust combination.

## Disadvantages

- **Limited to Linear Combinations**: The primary disadvantage is that it can only learn a linear relationship between the base models' predictions. If the optimal way to combine the models is non-linear (e.g., one model is better for low-risk predictions, while another is better for high-risk), Linear Regression will not be able to capture this and a more complex meta-model (like another tree-based model) might perform better.

- **Assumes No Multicollinearity**: Linear Regression assumes that the input features (in this case, the predictions from the base models) are not highly correlated. If two or more base models produce very similar predictions, it can make the coefficients of the meta-model unstable and harder to interpret.
