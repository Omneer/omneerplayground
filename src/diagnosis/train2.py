import pandas as pd
import numpy as np
from crankms import MLPClassifier as MLP
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'hidden_dims': [(32,), (64,), (128,)],  # Vary the number of hidden units and layers
    'num_epochs': [64, 128, 256],  # Vary the number of epochs
    'batch_size': [16, 32, 64],  # Vary the batch size
    'lambda_l1': [0.0001, 0.001, 0.01],  # Vary the L1 regularization parameter
    'lambda_l2': [0.0001, 0.001, 0.01]  # Vary the L2 regularization parameter
}


def train(
    model_name,
    dataset, train_size, valid_size, save_dir,
    seed=0,
):
    assert model_name == 'mlp'

    # Initialize the model
    model = MLP(
        hidden_dims=(32,),
        num_epochs=64,
        batch_size=16,
        lambda_l1=0.0011697870951761014,
        lambda_l2=0.0004719190005714674,
        device='cpu'
    )

    # Define the parameter search space
    param_grid = {
        'hidden_dims': [(32,), (64,), (128,)],
        'num_epochs': [64, 128, 256],
        'batch_size': [16, 32, 64],
        'lambda_l1': [0.0001, 0.001, 0.01],
        'lambda_l2': [0.0001, 0.001, 0.01]
    }

    # Create the search object
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring='accuracy',  # Use an appropriate evaluation metric
        n_iter=10,  # Adjust the number of iterations as desired
        cv=5,  # Adjust the number of cross-validation folds as desired
        random_state=seed
    )

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        *dataset.all,
        train_size=train_size,
        test_size=valid_size,
        random_state=seed,
    )

    # Normalize features
    norm_obj = StandardScaler().fit(x_train)
    x_train = norm_obj.transform(x_train)
    x_test = norm_obj.transform(x_test)

    # Convert labels to integers
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Perform parameter search
    search.fit(x_train, y_train)

    # Get the best parameters and score
    best_params = search.best_params_
    best_score = search.best_score_

    print("Best parameters:", best_params)
    print("Best score:", best_score)

    # Train the model with the best parameters
    model.set_params(**best_params)
    model.fit(x_train, y_train)

    # Save model checkpoint
    joblib.dump(model, '{}/checkpoints/test_{}.ckpt'.format(save_dir, seed))

    # Evaluate and save results
    s_test = model.predict_proba(x_test)[:, 1] if model_name != 'svm' else model.decision_function(x_test)
    p_test = model.predict(x_test)
    _save_results(y_test, p_test, s_test, save_dir, seed)


def _save_results(labels, predictions, scores, save_dir, seed):
    df_results = pd.DataFrame()
    df_results['Y Label'] = labels
    df_results['Y Predicted'] = predictions
    df_results['Predicted score'] = scores
    df_results.to_csv('{}/results/test_{}.csv'.format(save_dir, seed), index=False)
