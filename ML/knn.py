import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from parents import basicInvestClassifier

class KNNPredicter(basicInvestClassifier):
    def train_test_model(self, n_neighbors, class_weight, algorithm, leaf_size, p, metric, model_path: str):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='uniform',
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric
        )

        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, model_path)

        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_df = pd.DataFrame(self.y_pred, columns=['Prediction'])
        self.prediction_df = pd.concat([self.df.iloc[len(self.X_train):].reset_index(drop=True), 
                                   self.y_pred_df.reset_index(drop=True)], axis=1)
        print(classification_report(self.y_test, self.y_pred))
        return self.model


# Example usage
model = KNNPredicter(df=pd.read_csv(r"C:\trade_strategy1\Lyskovo\full_data.csv", index_col=0))

model.train_test_model(
    n_neighbors=10,  # Default number of neighbors
    class_weight={0:1, 1:1},  # Equal weight for all neighbors
    algorithm='kd_tree',  # Let the algorithm decide the best approach
    leaf_size=10,  # Default leaf size for tree-based algorithms
    p=2,  # Euclidean distance (Minkowski with p=2)
    metric='minkowski',  # Standard metric for KNN
    model_path=r"C:\trade_strategy1\Lyskovo\models\knnEntryPointsModel.pkl"
)

model.visual(path=r'C:\trade_strategy1\Lyskovo\charts\KNN_entry_points_visualization.html').show()
model.analytycs()