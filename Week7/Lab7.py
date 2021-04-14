import pandas as pd

df = pd.read_csv('./healthcare-dataset-stroke-data.csv')

print(df.isnull().sum())

df.bmi.fillna(df.bmi.mean(), inplace=True) #Only feature with null values

df.smoking_status = df.smoking_status.replace({"Unknown": df.smoking_status.mode()[0]}) #Replacing Unknown with NaN

X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
        'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']]


