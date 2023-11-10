import pandas as pd
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, filepath):
        self.data = self.load_data(filepath)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    @staticmethod
    def load_data(filepath):
        return pd.read_csv(filepath, delimiter=';')

    def train_test_split(self, test_size=0.2, random_state=42):
        # x = self.data.drop('cardio', axis=1)
        x = self.data.drop(columns=['cardio', 'id'])
        y = self.data['cardio']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
                                                                                random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test
