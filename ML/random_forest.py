import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from parents import basicInvestClassifier

class RandomForestPredicter(basicInvestClassifier):
    def train_test_model(self, n_estimators, criterion,
                         max_depth, min_samples_split,
                         min_samples_leaf, max_features,
                         oob_score, random_state, class_weight,
                         model_path: str):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            oob_score=oob_score,
            random_state=random_state,
            class_weight=class_weight)

        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, model_path)
        self.y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, self.y_pred))
        self.y_pred_df = pd.DataFrame(self.y_pred, columns=['Prediction'])
        self.prediction_df = pd.concat([self.df.iloc[len(self.X_train):].reset_index(drop=True), 
                                   self.y_pred_df.reset_index(drop=True)], axis=1)
        return self.model

model = RandomForestPredicter(df=pd.read_csv(r"C:\trade_strategy1\Lyskovo\full_data.csv", index_col=0))
model.train_test_model(n_estimators=50,
    criterion='gini',
    max_depth=5,
    min_samples_split=15,
    min_samples_leaf=40,
    max_features='log2',
    class_weight={0: 1, 1: 1.1},
    oob_score=True,
    random_state=None,
    model_path=r"C:\trade_strategy1\Lyskovo\models\randomForestEntryPointsModel.pkl")

model.visual(path=r'C:\trade_strategy1\Lyskovo\charts\RF_entry_points_visualization.html')
model.analytycs()