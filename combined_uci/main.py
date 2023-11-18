from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report, f1_score
import matplotlib.pyplot as plt
from data_handler import DataHandler
from models import XGBoostModel, LRModel, KNNModel, RandomForest


def main():
    handler = DataHandler('../data/cleveland_new.csv')
    x_train, x_test, y_train, y_test = handler.train_test_split()

    print("XGBoost:")
    XGModelMain = XGBoostModel(x_train, x_test, y_train, y_test)
    best_xgb, score_xgb = XGModelMain.hyperparameter_tuning()
    XGModelMain.test_best_model(best_xgb)

    print("Logistic Regression:")
    LRModelMain = LRModel(x_train, x_test, y_train, y_test)
    best_lr, score_lr = LRModelMain.hyperparameter_tuning()
    LRModelMain.test_best_model(best_lr)

    print("KNN:")
    KNNModelMain = KNNModel(x_train, x_test, y_train, y_test)
    best_knn, score_knn = KNNModelMain.hyperparameter_tuning()
    KNNModelMain.test_best_model(best_knn)

    print("Random forest:")
    RFModelMain = RandomForest(x_train, x_test, y_train, y_test)
    best_rf, score_rf = RFModelMain.hyperparameter_tuning()
    RFModelMain.test_best_model(best_rf)

    # Create and evaluate the weighted ensemble
    print("Creating Weighted Ensemble:")
    ensemble = create_weighted_ensemble(x_train, y_train, best_xgb, best_lr, best_knn, best_rf, score_xgb, score_lr,
                                        score_knn, score_rf)
    ensemble_predictions = ensemble.predict(x_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    ensemble_f1 = f1_score(y_test, ensemble_predictions, average='binary')
    ensemble_conf = confusion_matrix(y_test, ensemble_predictions)
    ensemble_report = classification_report(y_test, ensemble_predictions)
    y_pred_prob = ensemble.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    ensemble_roc_auc = roc_auc_score(y_test, y_pred_prob)
    print('Scores on Weighted Ensemble: ')
    print(f'Ensemble Accuracy: {ensemble_accuracy:.4f}')
    print(f'F1 Score: {ensemble_f1:.4f}')
    print(f'ROC Score: {ensemble_roc_auc:.4f}')
    print(f'Confusion matrix : \n{ensemble_conf}')
    print(f'Classification report: \n{ensemble_report}')

    # Plot ROC curve
    '''plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linestyle='-', linewidth=2, label=f'ROC Curve (AUC = {ensemble_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()'''

    print("Creating Stacked Ensemble:")
    stacked_ensemble = create_stacked_ensemble(x_train, y_train, best_xgb, best_lr, best_knn, best_rf)
    stacked_ensemble_predictions = stacked_ensemble.predict(x_test)
    stacked_ensemble_accuracy = accuracy_score(y_test, stacked_ensemble_predictions)
    stacked_ensemble_f1 = f1_score(y_test, stacked_ensemble_predictions, average='binary')
    stacked_ensemble_conf = confusion_matrix(y_test, stacked_ensemble_predictions)
    stacked_ensemble_report = classification_report(y_test, stacked_ensemble_predictions)
    stacked_y_pred_prob = stacked_ensemble.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, stacked_y_pred_prob)
    stacked_ensemble_roc_auc = roc_auc_score(y_test, stacked_y_pred_prob)
    print('Scores on Stacked Ensemble: ')
    print(f'Ensemble Accuracy: {stacked_ensemble_accuracy:.4f}')
    print(f'F1 Score: {stacked_ensemble_f1:.4f}')
    print(f'ROC Score: {stacked_ensemble_roc_auc:.4f}')
    print(f'Confusion matrix : \n{stacked_ensemble_conf}')
    print(f'Classification report: \n{stacked_ensemble_report}')


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


def create_stacked_ensemble(x_train, y_train, best_xgb, best_lr, best_knn, best_rf):
    # Define base models
    base_models = [
        ('xgb', best_xgb),
        ('lr', best_lr),
        ('knn', best_knn),
        ('rf', best_rf)
    ]

    # Choose a meta-model
    meta_model = LogisticRegression()

    # Create the stacked ensemble
    stacked_ensemble = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

    # Fit the ensemble on the training data
    stacked_ensemble.fit(x_train, y_train)

    return stacked_ensemble


if __name__ == "__main__":
    main()
