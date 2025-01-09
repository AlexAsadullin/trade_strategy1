from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib

from parents import basicInvestClassifier

class GradientBoostingPredicter(basicInvestClassifier):
    def train_test_model(self, learning_rate, n_estimators, subsample,
                         criterion, min_samples_split, min_samples_leaf,
                         max_depth, random_state, class_weight,
                         model_path: str):
        self.model = GradientBoostingClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=random_state
        )

        sample_weights = self.y_train.map(class_weight)
        self.model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        joblib.dump(self.model, model_path)
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_df = pd.DataFrame(self.y_pred, columns=['Prediction'])
        self.prediction_df = pd.concat([self.df.iloc[len(self.X_train):].reset_index(drop=True), 
                                   self.y_pred_df.reset_index(drop=True)], axis=1)
        print(classification_report(self.y_test, self.y_pred))
        return self.model

def main(df_path:str, model_path:str, chart_path:str):
    model = GradientBoostingPredicter(df=pd.read_csv(df_path, index_col=0))
    model.train_test_model(
        learning_rate = 0.1,
        n_estimators = 100,
        subsample = 0.8,
        criterion = 'friedman_mse',
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_depth = 3,
        random_state = 42,
        class_weight={0:2, 1:5},
        model_path=model_path)

    model.visual(path=chart_path)
    model.analytycs()

if __name__ == '__main__':
    main(df_path=r"/home/alex/BitcoinScalper/dataframes/full_data.csv",
        model_path=r"/home/alex/BitcoinScalper/ML/models/gradientBoostingEntryPointsModel.pkl",
        chart_path=r'/home/alex/BitcoinScalper/charts/GB_entry_points_visualization.html')