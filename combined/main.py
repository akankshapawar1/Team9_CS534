from data_handler import DataHandler
from models import XGBoostModel, LRModel, KNNModel
import pandas as pd


def main():
    handler = DataHandler('/Users/akanksha/Desktop/534/GP/Team9_CS534/data/cardio_train.csv')
    x_train, x_test, y_train, y_test = handler.train_test_split()

    print("XGBoost")
    XGModelMain = XGBoostModel(x_train, y_train)
    XGModelMain.hyperparameter_tuning()
    XGModelMain.test_best_model(x_test, y_test)

    print("Logistic Regression")
    LRModelMain = LRModel(x_train, x_test, y_train, y_test)
    LRModelMain.hyper_parameter_tuning()
    LRModelMain.test_best_model()

    print("KNN")
    KNNModelMain = KNNModel(x_train, x_test, y_train, y_test)
    KNNModelMain.hyper_parameter_tuning()
    KNNModelMain.test_best_model()


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