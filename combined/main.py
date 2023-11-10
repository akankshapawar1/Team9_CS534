from data_handler import DataHandler
from models import XGBoostModel, LRModel, KNNModel


def main():
    handler = DataHandler('/Users/akanksha/Desktop/534/GP/Team9_CS534/data/cardio_train.csv')
    x_train, x_test, y_train, y_test = handler.train_test_split()

    print("XGBoost")
    XGModelMain = XGBoostModel(handler.data.drop(columns=['cardio','id']), handler.data['cardio'])
    XGModelMain.cross_validate(k=10)

    print("Logistic Regression")
    LRModelMain = LRModel(x_train, x_test, y_train, y_test)
    LRModelMain.train()

    print("KNN")
    KNNModelMain = KNNModel(x_train, x_test, y_train, y_test)
    KNNModelMain.find_best_parameter()


if __name__ == "__main__":
    main()
