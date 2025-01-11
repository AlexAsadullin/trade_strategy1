import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer

class basicInvestClassifier:
    def __init__(self, df):
        self.df = df
        self._load_prepare_data()

    def _load_prepare_data(self):
        self.df = self.df.drop(['Open', 'High','Low', 'Time', 'AvgOpenClose','DiffOpenClose','DiffHighLow'], axis=1)
        # prepare data
        """self.imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(self.imputer.fit_transform(self.df), columns=self.df.columns)"""
        scaler = Normalizer()
        X = self.df.drop('IsEntryPoint', axis=1)
        y = self.df['IsEntryPoint']

        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    def analytycs(self):
        self.prediction_df = pd.concat([self.df.iloc[len(self.X_train):].reset_index(drop=True), 
                                   self.y_pred_df.reset_index(drop=True)], axis=1)
        entry_points = len(self.prediction_df[self.prediction_df['IsEntryPoint'] == 1])
        true_positive = len(self.prediction_df[(self.prediction_df['IsEntryPoint'] == 1) & (self.prediction_df['Prediction'] == 1)])
        false_positive = len(self.prediction_df[(self.prediction_df['IsEntryPoint'] == 0) & (self.prediction_df['Prediction'] == 1)])
        true_negative = len(self.prediction_df[(self.prediction_df['IsEntryPoint'] == 0) & (self.prediction_df['Prediction'] == 0)])
        false_negative = len(self.prediction_df[(self.prediction_df['IsEntryPoint'] == 1) & (self.prediction_df['Prediction'] == 0)])
        total_matches = len(self.prediction_df[self.prediction_df['IsEntryPoint'] == self.prediction_df['Prediction']])
        all_rows = len(self.prediction_df)
        print('entry points:', entry_points)
        print('true positive:', true_positive)
        print('false positive:', false_positive)

        print('true negative:', true_negative)
        print('false negative:', false_negative)

        print('total matches:', total_matches)
        print('all rows:', all_rows)
        print(self.prediction_df.head())
        return {
            'entry_points': entry_points,
            'true_positive': true_positive,
            'false_positive': false_positive,
            'true_negative': true_negative,
            'false_negative': false_negative,
            'total_matches': total_matches,
            'all_rows': all_rows,
        }

    def visual(self, path:str):
        real_points = self.prediction_df[self.prediction_df['IsEntryPoint'] == 1]
        predicted_points = self.prediction_df[self.prediction_df['Prediction'] == 1]
        fig = go.Figure()
        # Close price
        fig.add_trace(go.Scatter(
            x=self.prediction_df.index, y=self.prediction_df["Close"], mode='lines',
            line=dict(color="blue"), name='Close'))
        # real entry points & predicted entry points
        fig.add_trace(go.Scatter(x=real_points.index,
            y=real_points['Close'], mode='markers', marker=dict(color='purple', size=7), name='Entry Points'))
        fig.add_trace(go.Scatter(x=predicted_points.index,
            y=predicted_points['Close'], mode='markers', marker=dict(color='orange', size=4), name='Predicted Entry Points'))
        # min & max
        fig.add_trace(go.Scatter(
            x=self.prediction_df.index, y=self.prediction_df["Max"], mode='lines', marker=dict(color='green', size=0.8), name='Max Points'))
        fig.add_trace(go.Scatter(
            x=self.prediction_df.index, y=self.prediction_df["Min"], mode='lines', marker=dict(color='red', size=0.8), name='Min Points'))
        fig.write_html(path)
        return fig