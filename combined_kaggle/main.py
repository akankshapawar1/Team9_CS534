from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from data_handler import DataHandler
from models import XGBoostModel, LRModel, KNNModel, RandomForest


def main():
    handler = DataHandler('/data/kaggle/cardio_train.csv')
    x_train, x_test, y_train, y_test = handler.train_test_split()

    print("XGBoost")
    XGModelMain = XGBoostModel(x_train, y_train)
    best_xgb, score_xgb = XGModelMain.hyperparameter_tuning()
    # XGModelMain.test_best_model(x_test, y_test)

    print("Logistic Regression")
    LRModelMain = LRModel(x_train, x_test, y_train, y_test)
    best_lr, score_lr = LRModelMain.hyper_parameter_tuning()
    # LRModelMain.test_best_model()

    print("KNN")
    KNNModelMain = KNNModel(x_train, x_test, y_train, y_test)
    best_knn, score_knn = KNNModelMain.hyper_parameter_tuning()
    # KNNModelMain.test_best_model()

    print("Random forest")
    RFModelMain = RandomForest(x_train, x_test, y_train, y_test)
    best_rf, score_rf = RFModelMain.hyper_parameter_tuning()
    # RFModelMain.test_best_model()

    # Create and evaluate the weighted ensemble
    print("Creating Weighted Ensemble")
    ensemble = create_weighted_ensemble(x_train, y_train, best_xgb, best_lr, best_knn, best_rf, score_xgb, score_lr,
                                        score_knn, score_rf)
    ensemble_predictions = ensemble.predict(x_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    print(f'Ensemble Accuracy: {ensemble_accuracy:.4f}')


def create_weighted_ensemble(x_train, y_train, best_xgb, best_lr, best_knn, best_rf, score_xgb, score_lr,
                             score_knn, score_rf):
    # Use provided best models and their scores to create ensemble
    models = [
        ('xgb', best_xgb, score_xgb),
        ('lr', best_lr, score_lr),
        ('knn', best_knn, score_knn),
        ('rf', best_rf, score_rf)
    ]

    total_score = sum(score for _, _, score in models)
    weights = [score / total_score for _, _, score in models]

    ensemble = VotingClassifier(estimators=[(name, model) for name, model, _ in models], voting='soft', weights=weights)
    ensemble.fit(x_train, y_train)
    return ensemble


if __name__ == "__main__":
    main()

'''
results- 
XGBoost (hyper)
Fitting 10 folds for each of 36 candidates, totalling 360 fits
Best accuracy: 0.7371285714285714 using {'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.8}
Logistic Regression
Accuracy: 0.71
f1_score: 0.70
KNN
Best number of neighbors: 39 with F1 Score: 0.7150

--------------------------------------------------------------- testing on best models

XGBoost
Fitting 10 folds for each of 36 candidates, totalling 360 fits
Best accuracy: 0.735625 using {'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.8}
Best model saved as best_model.pkl
Accuracy on test with best paras : 0.7403
F1 Score on test with best paras : 0.7296

XGBoost
Fitting 10 folds for each of 36 candidates, totalling 360 fits
Best accuracy: 0.7357678571428571 using {'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8}
Best model saved as best_model.pkl
Accuracy on test with best paras : 0.7401
F1 Score on test with best paras : 0.7285

Logistic Regression
Accuracy on test with best paras : 0.7228
F1 Score on test with best paras : 0.7065

KNN
Best F1 Score: 0.6981 with parameters: {'algorithm': 'auto', 'n_neighbors': 10, 'p': 1, 'weights': 'uniform'}
Best model saved as best_knn_model.pkl
Accuracy on test with best paras : 0.7008
F1 Score on test with best paras : 0.6748

Best F1 Score: 0.6850 with parameters: {'algorithm': 'auto', 'n_neighbors': 9, 'p': 2, 'weights': 'uniform'}
Best model saved as best_knn_model.pkl
Accuracy on test with best paras : 0.7009
F1 Score on test with best paras : 0.6943

Random forest
Fitting 10 folds for each of 180 candidates, totalling 1800 fits
Best F1 Score: 0.7350 with parameters: {'max_depth': 10, 'min_samples_leaf': 10, 'n_estimators': 100}
Creating Weighted Ensemble
Ensemble Accuracy: 0.7371

'''

'''for normalized data
    handler = DataHandler('/Users/akanksha/Desktop/534/GP/Team9_CS534/cardio_cleaned.csv')
    # LR - Accuracy: 0.59 f1_score: 0.46
    # xgboost - Average F1 Score: 0.5760
    x_train, x_test, y_train, y_test = handler.train_test_split()
    
    # XGModelMain = XGBoostModel(handler.data.drop(columns=['cardio']), handler.data['cardio'])
    
    def train_test_split(self, test_size=0.2, random_state=42):
        # x = self.data.drop('cardio', axis=1)
        y = self.data['cardio']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
                                                                                random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def load_data(filepath):
        df = pd.read_csv(filepath, delimiter=',')
        df = df.dropna()
        print(df.shape[0])
        return df
'''
