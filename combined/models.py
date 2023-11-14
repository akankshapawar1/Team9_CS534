import joblib
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np
from joblib import load
import pandas as pd


class XGBoostModel:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def cross_validate(self, k=10):
        model = XGBClassifier()
        '''scores = cross_val_score(model, self.x_data, self.y_data, cv=k, scoring='accuracy', n_jobs=-1)
        for i, score in enumerate(tqdm(scores, desc="Cross-validation", unit="fold")):
            print(f"Fold {i + 1} Accuracy: {score:.4f}")
        print(f"Average Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")'''

        f1_scores = cross_val_score(model, self.x_data, self.y_data, cv=k, scoring='f1_macro', n_jobs=-1)
        for i, score in enumerate(tqdm(f1_scores, desc="F1 Score Cross-validation", unit="fold")):
            print(f"Fold {i + 1} F1 Score: {score:.4f}")
        print(f"Average F1 Score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")

    def hyperparameter_tuning(self, k=10):
        param_grid = {
            'max_depth': [3, 4, 5],
            # 'max_depth': [5, 6, 7],
            'subsample': [0.6, 0.8, 1.0],
            'learning_rate': [0.01, 0.02, 0.05, 0.1]
        }
        model = XGBClassifier()
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'f1_score': make_scorer(f1_score)}
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   scoring=scoring, refit='accuracy', cv=k, verbose=1, n_jobs=-1)
        grid_search.fit(self.x_data, self.y_data)
        print(f"Best accuracy: {grid_search.best_score_} using {grid_search.best_params_}")
        # return grid_search.best_estimator_

        # Save the best model
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        '''joblib.dump(best_model, 'best_xgboost_model.pkl')
        print("Best model saved as best_xgboost_model.pkl")'''

        return best_model, best_score

    def test_best_model(self, x_test, y_test):
        model_path = 'best_xgboost_model.pkl'
        loaded_model = load(model_path)
        predictions = loaded_model.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy on test with best paras : {accuracy:.4f}')

        f1 = f1_score(y_test, predictions, average='binary')  # Use 'binary' for binary classification
        print(f'F1 Score on test with best paras : {f1:.4f}')


class LRModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def hyper_parameter_tuning(self, k=10):
        param_grid = {
            'solver': ['liblinear'],
            'penalty': ['l2'],
            'C': [100, 10, 1.0, 0.1, 0.01]
        }
        logreg = LogisticRegression()
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'f1_score': make_scorer(f1_score)}
        grid_search = GridSearchCV(logreg, param_grid, scoring=scoring, refit='accuracy', cv=k, verbose=1, n_jobs=-1)
        # grid_search.fit(self.x_train[['age', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'active']],
        # self.y_train)
        grid_search.fit(self.x_train, self.y_train)

        print(f"Best accuracy: {grid_search.best_score_} using {grid_search.best_params_}")

        # Save the best model
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        '''joblib.dump(best_model, 'best_logistic_regression_model.pkl')
        print("Best model saved as best_logistic_regression_model.pkl")'''

        return best_model, best_score

    def test_best_model(self):
        model_path = 'best_logistic_regression_model.pkl'
        loaded_model = load(model_path)
        # predictions = loaded_model.predict(self.x_test[['age', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke',
        # 'active']])
        predictions = loaded_model.predict(self.x_test)

        accuracy = accuracy_score(self.y_test, predictions)
        print(f'Accuracy on test with best paras : {accuracy:.4f}')

        f1 = f1_score(self.y_test, predictions, average='binary')
        print(f'F1 Score on test with best paras : {f1:.4f}')


class KNNModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def hyper_parameter_tuning(self, average_method='binary'):
        param_grid = {'n_neighbors': np.arange(1, 11),
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'p': [1, 2]}

        knn = KNeighborsClassifier()

        grid_search = GridSearchCV(knn, param_grid, scoring=make_scorer(f1_score, average=average_method), cv=10,
                                 verbose=1, n_jobs=-1)
        '''scoring = {'accuracy': make_scorer(accuracy_score),
                   'f1_score': make_scorer(f1_score, average=average_method, zero_division=1)}
        grid_search = GridSearchCV(knn, param_grid, scoring=scoring, refit='f1_score', cv=10, verbose=1, n_jobs=-1, error_score='raise')'''

        grid_search.fit(self.x_train, self.y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best F1 Score: {best_score:.4f} with parameters: {best_params}")

        # Save the best model
        '''joblib.dump(best_model, 'best_knn_model.pkl')
        print("Best model saved as best_knn_model.pkl")'''

        return best_model, best_score

    def test_best_model(self):
        model_path = 'best_knn_model.pkl'
        loaded_model = load(model_path)
        predictions = loaded_model.predict(self.x_test)

        accuracy = accuracy_score(self.y_test, predictions)
        print(f'Accuracy on test with best paras : {accuracy:.4f}')

        f1 = f1_score(self.y_test, predictions, average='binary')
        print(f'F1 Score on test with best paras : {f1:.4f}')


class RandomForest:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def hyper_parameter_tuning(self):
        params = {
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100, 200],
            'n_estimators': [10, 25, 30, 50, 100, 200]
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(estimator=rf,
                                   param_grid=params,
                                   cv=10,
                                   n_jobs=-1, verbose=1, scoring="accuracy")

        grid_search.fit(self.x_train, self.y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best F1 Score: {best_score:.4f} with parameters: {best_params}")

        # Save the best model
        '''joblib.dump(best_model, 'best_rf_model.pkl')
        print("Best model saved as best_rf_model.pkl")'''

        return best_model, best_score

    def test_best_model(self):
        model_path = 'best_rf_model.pkl'
        loaded_model = load(model_path)
        predictions = loaded_model.predict(self.x_test)

        accuracy = accuracy_score(self.y_test, predictions)
        print(f'Accuracy on test with best paras : {accuracy:.4f}')

        f1 = f1_score(self.y_test, predictions, average='binary')
        print(f'F1 Score on test with best paras : {f1:.4f}')
