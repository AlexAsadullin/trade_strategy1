import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from parents import basicInvestClassifier

class SVMModel(basicInvestClassifier):
    def train_test_model(self, kernel, C, degree, gamma, coef0, class_weight, probability, random_state, model_path: str):
        self.model = SVC(
            kernel=kernel,
            C=C,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state
        )

        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, model_path)

        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_df = pd.DataFrame(self.y_pred, columns=['Prediction'])
        self.prediction_df = pd.concat([self.df.iloc[len(self.X_train):].reset_index(drop=True), 
                                   self.y_pred_df.reset_index(drop=True)], axis=1)
        print(classification_report(self.y_test, self.y_pred))
        return self.model

model = SVMModel(df=pd.read_csv(r"C:\trade_strategy1\Lyskovo\full_data.csv", index_col=0))
model.train_test_model(
    kernel='rbf',
    C=1.0,
    degree=3,
    gamma='scale',
    coef0=0.0,
    class_weight={0: 2, 1: 5},
    probability=True,
    random_state=42,
    model_path=r"C:\trade_strategy1\Lyskovo\models\SVMEntryPointsModel.pkl"
)

model.visual(path=r'C:\trade_strategy1\Lyskovo\charts\SVM_entry_points_visualization.html')
model.analytycs()