from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import metrics


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
            'subsample': [0.6, 0.8, 1.0],
            'learning_rate': [0.01, 0.02, 0.05, 0.1]
        }
        model = XGBClassifier()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   scoring='accuracy', cv=k, verbose=1, n_jobs=-1)
        grid_search.fit(self.x_data, self.y_data)
        print(f"Best accuracy: {grid_search.best_score_} using {grid_search.best_params_}")
        return grid_search.best_estimator_


class KNNModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self, average_method='macro'):
        f1_scores = {}
        for i in range(1, 101):
            knn_classifier = KNeighborsClassifier(n_neighbors=i)
            knn_classifier.fit(self.x_train, self.y_train)
            predictions = knn_classifier.predict(self.x_test)
            f1 = f1_score(self.y_test, predictions, average=average_method)
            f1_scores[i] = f1
            # print(f'Neighbors: {i}, F1 Score: {f1:.4f}')
        return f1_scores

    def find_best_parameter(self, average_method='macro'):
        f1_scores = self.train(average_method)
        best_n = max(f1_scores, key=f1_scores.get)
        print(f'Best number of neighbors: {best_n} with F1 Score: {f1_scores[best_n]:.4f}')
        return best_n


class LRModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        logreg = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
        # logreg.fit(self.x_train[['ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'active']], self.y_train)
        # y_pred = logreg.predict(self.x_test[['ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'active']])
        # Accuracy: 0.72 f1_score: 0.69
        logreg.fit(self.x_train, self.y_train)
        y_pred = logreg.predict(self.x_test)
        # Accuracy: 0.71 f1_score: 0.70
        print('Accuracy: %.2f' % metrics.accuracy_score(self.y_test, y_pred))
        print('f1_score: %.2f' % metrics.f1_score(self.y_test, y_pred))
