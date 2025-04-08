from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

import numpy as np
import pandas as pd
from itertools import product


randome_state = 42

def run_pca(df, labels_dict = None, n_components=2, drop_cols=['pursuit_task_id'], top_n_features=3):
    '''
    runs pca, returns important metrics and the resulting dataframe

    '''

    # scale the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(columns=drop_cols).values)
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(pca_components,
                          columns=[f"PC{i+1}" for i in range(n_components)],
                          index=df.index)
    # add back the pursuit task id
    pca_df['pursuit_task_id'] = df['pursuit_task_id']
    # adds labels where ther are labels
    if labels_dict is not None:
        pca_df['label'] = pca_df['pursuit_task_id'].map(labels_dict)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pca_loadings = pd.DataFrame(loadings,
                                columns=[f"PC{i+1}" for i in range(n_components)])
    
    top_features = {}
    for i in range(n_components):
        top_feats = pca_loadings.iloc[i].abs().nlargest(top_n_features)
        top_features[f"PC{i+1}"] = top_feats
    
    result = {"pca_df": pca_df,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance_ratio": cumulative_variance_ratio,
        "pca_loadings": pca_loadings,
        "top_features": top_features}

    return result




def train_logistic_models(X, y, cv_folds=10):
    """
    Trains a Logistic Regression model for each fold in Stratified K-Fold cross-validation.
    Parameters:
    X (pd.DataFrame): Features for the model.
    y (pd.Series): Target labels.
    cv_folds (int): Number of folds for cross-validation (default is 10).
    Returns:
    list: List of trained models, one for each fold.
    """

    # does scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=randome_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='accuracy')
    estimated_accuracy = np.mean(cv_scores)
    print(f"Estimated accuracy from {cv_folds}-fold CV: {estimated_accuracy}")

    pipeline.fit(X, y)
    return pipeline 

def train_xgb_cv(X, y, n_trees, max_depths, learning_rates, cv_folds = 10):
    """
    trains multiple xgboost models with different hyperparamaters and returns the best one
    """
    class_labels = np.unique(y)

    params_to_test = [{"n_estimators": nt, "max_depth": md, "learning_rate": lr}
                      for nt, md, lr in product(n_trees, max_depths, learning_rates)]
    
    best_score = 0
    best_params = None

    for params in params_to_test:
        # train with these params
        xgb = XGBClassifier(**params, eval_metric="logloss", random_state=randome_state)
        test_accuracies = cross_val_score(xgb, X, y, cv=cv_folds, scoring='accuracy')
        mean_test_accuracies = np.mean(test_accuracies)

        if mean_test_accuracies > best_score:
            best_score = mean_test_accuracies
            best_params = params
    print(f"Best params for xgboost model: {best_params} with accuracy: {best_score:.3f}")

    # train the model with best params
    best_model = XGBClassifier(**best_params, eval_metric="logloss", random_state=randome_state)
    best_model.fit(X, y)

    return best_model


def train_rf_cv(X, y, n_trees, max_depths, min_samples_splits, cv_folds=10):
    """
    trains multiple random forest models with different hyperparamaters and returns the best one
    """
    # max features at each split auto set to sqrt features and that is fine
    params_to_test = [{"n_estimators": nt, "max_depth": md, "min_samples_split": mss}
                      for nt, md, mss in product(n_trees, max_depths, min_samples_splits)]
    

    best_score = 0
    best_params = None

    for params in params_to_test:
        # train with these params
        rf = RandomForestClassifier(**params, random_state=randome_state)
        test_accuracies = cross_val_score(rf, X, y, cv=cv_folds, scoring='accuracy')
        mean_test_accuracies = np.mean(test_accuracies)

        if mean_test_accuracies > best_score:
            best_score = mean_test_accuracies
            best_params = params
    
    print(f"Best params for random forest model: {best_params} with accuracy: {best_score:.3f}")

    # train the model with best params
    best_model = RandomForestClassifier(**best_params, random_state=randome_state)
    best_model.fit(X, y)

    return best_model

def train_knn_cv(X, y, neighbor_vals, weight_options=['uniform', 'distance'], cv_folds=10):
    '''
    trains multiple knn models with different hyperparamaters and returns the best one
    '''
    params_to_test = [{"n_neighbors": k, "weights": w} for k, w in product(neighbor_vals, weight_options)]

    best_score = 0
    best_params = None

    for params in params_to_test:
        knn = KNeighborsClassifier(**params)
        test_accuracies = cross_val_score(knn, X, y, cv=cv_folds, scoring='accuracy')
        mean_test_accuracies = np.mean(test_accuracies)

        if mean_test_accuracies > best_score:
            best_score = mean_test_accuracies
            best_params = params
    
    print(f"Best params for KNN model: {best_params} with accuracy: {best_score:.3f}")

    # train the model with best params
    best_model = KNeighborsClassifier(**best_params)
    best_model.fit(X, y)

    return best_model