import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from ml_parents import basicInvestClassifier

class MLPModel(basicInvestClassifier):
    def train_test_model(self, hidden_layer_sizes, activation, solver,
                         alpha, batch_size, learning_rate,
                         learning_rate_init, max_iter, random_state,
                         model_path: str):
        """
        Train and test the MLP model.
        
        Parameters:
        - hidden_layer_sizes: Tuple defining the number of neurons in each hidden layer.
        - activation: Activation function for the hidden layer ('relu', 'logistic', etc.).
        - solver: Solver for weight optimization ('adam', 'sgd', etc.).
        - alpha: L2 penalty (regularization term).
        - batch_size: Size of minibatches for stochastic optimizers.
        - learning_rate: Learning rate schedule ('constant', 'invscaling', 'adaptive').
        - learning_rate_init: Initial learning rate.
        - max_iter: Maximum number of iterations.
        - random_state: Seed for random number generation.
        - model_path: Path to save the trained model.
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
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

def main(df_path:str, model_path:str, chart_path:str):
    model = MLPModel(df=pd.read_csv(df_path, index_col=0))
    model.train_test_model(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        model_path=model_path
    )

    model.visual(path=chart_path)
    model.analytycs()

if __name__ == '__main__':
    main(df_path=r"/home/alex/BitcoinScalper/dataframes/full_data.csv",
        model_path=r"/home/alex/BitcoinScalper/ML/models/MLP_EntryPointsModel.pkl",
        chart_path=r'/home/alex/BitcoinScalper/charts/MLP_entry_points_visualization.html')