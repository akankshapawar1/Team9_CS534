import pandas as pd
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, filepath):
        self.data = self.load_data(filepath)
        self.processed_df = self.preprocess()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    @staticmethod
    def load_data(filepath):
        df = pd.read_csv(filepath, delimiter=',')
        return df

    def preprocess(self):
        self.data.columns = ['age', 'sex', 'chest_pain', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                             'rest_ecg', 'max_heart_rate', 'exercise_induced_angina', 'st_depression', 'st_slope_type',
                             'num_major_vessels', 'thalassemia', 'target']
        df = self.data
        # print('Columns before: ', df.columns)
        # cp - chest_pain
        df.loc[df['chest_pain'] == 0, 'chest_pain'] = 'asymptomatic'
        df.loc[df['chest_pain'] == 1, 'chest_pain'] = 'atypical_angina'
        df.loc[df['chest_pain'] == 2, 'chest_pain'] = 'non-anginal_pain'
        df.loc[df['chest_pain'] == 3, 'chest_pain'] = 'typical_angina'

        # restecg - rest_ecg
        df.loc[df['rest_ecg'] == 0, 'rest_ecg'] = 'left_ventricular_hypertrophy'
        df.loc[df['rest_ecg'] == 1, 'rest_ecg'] = 'normal'
        df.loc[df['rest_ecg'] == 2, 'rest_ecg'] = 'STT_abnormality'

        # slope - st_slope_type
        df.loc[df['st_slope_type'] == 0, 'st_slope_type'] = 'downsloping'
        df.loc[df['st_slope_type'] == 1, 'st_slope_type'] = 'flat'
        df.loc[df['st_slope_type'] == 2, 'st_slope_type'] = 'upsloping'

        # thal - thalassemia
        df.loc[df['thalassemia'] == 0, 'thalassemia'] = 'nothing'
        df.loc[df['thalassemia'] == 1, 'thalassemia'] = 'fixed'
        df.loc[df['thalassemia'] == 2, 'thalassemia'] = 'normal'
        df.loc[df['thalassemia'] == 3, 'thalassemia'] = 'reversible'

        df = pd.get_dummies(df, drop_first=False)
        df = df.astype(int)
        '''print(df.head())
        print('Columns after: ', df.columns)'''
        return df

    def train_test_split(self, test_size=0.2, random_state=42):
        x = self.processed_df.drop(columns=['target'])
        y = self.processed_df['target']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
                                                                                random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test
