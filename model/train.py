from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import pathlib

data = pd.read_csv(pathlib.Path('data/personality.csv'))

print(data.columns)

y = LabelEncoder().fit_transform(data["Personality"])

X = data[[
    "Age", "Education", "IntroversionScore", "SensingScore", 
    "ThinkingScore", "JudgingScore", "Gender", "Interest"
]]

categorical_features = ["Gender", "Interest"]
numeric_features = ["Age", "Education", "IntroversionScore", "SensingScore", "ThinkingScore", "JudgingScore"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('Training model..')
pipeline.fit(X_train, y_train)

print('Saving model..')
dump(pipeline, pathlib.Path('model/personality-pipeline-v1.joblib'))
