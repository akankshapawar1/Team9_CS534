import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# load data into a Pandas DataFrame
df = pd.read_csv('/data/cardio_train.csv', sep=';')

#patients_with_height_less_than_121 = df[df['height'] < 121].shape[0]
#print(patients_with_height_less_than_121)

#patients_with_height_less_than_213 = df[df['height'] > 200].shape[0]
#print(patients_with_height_less_than_213)

df = df[df['height'] >= 121]

df['gender'] -= 1  # convert gender values to be either 1 or 0
df['age'] //= 356

df = df[df['ap_hi'] < 250]
df = df[df['ap_lo'] < 250]

#print(df.shape[0])
#print(df.head())

cont_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
ord_cols = ['cholesterol', 'gluc']
bin_cols = ['gender', 'smoke', 'alco', 'active', 'cardio']
# create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()
# create an object to transform the data to fit minmax processor
df_scaled = min_max_scaler.fit_transform(df[cont_cols])

# run the normalizer on the dataframe
df_normalized = pd.DataFrame(df_scaled)
df_normalized.columns = cont_cols
#print(df_normalized.head())

df_normalized = pd.concat([df_normalized, df[ord_cols], df[bin_cols]], axis=1)

# print(df_normalized.head())

# save this cleaned data
# df_normalized.to_csv('cardio_cleaned.csv', index=False)

# print(df_normalized.columns)

'''def check_outliers(df):
    column_names = df.columns.values
    l = df.columns.values
    number_of_columns = int((len(l) - 1) / 2)
    number_of_rows = 2

    # Adjust the figure size here as needed
    plt.figure(figsize=(5 * number_of_columns, 10 * number_of_rows))
    for i in range(1, len(l)):
        plt.subplot(number_of_rows + 1, number_of_columns, i)
        sns.set_style('whitegrid')
        sns.boxplot(df[l[i]], orient='v')
        plt.xlabel(column_names[i])
        # Optionally adjust subplot parameters manually
        plt.subplots_adjust(hspace=0.4, wspace=0.4)


check_outliers(df)
plt.show()'''


