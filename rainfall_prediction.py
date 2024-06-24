import opendatasets as od
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



dataset_url = 'https://www.kaggle.com/jsphyg/weather-dataset-rattle-package'
#od.download(dataset_url)

data_dir = './weather-dataset-rattle-package'
os.listdir(data_dir)
train_csv = data_dir + '/weatherAUS.csv'

raw_df = pd.read_csv(train_csv)
#print(raw_df.info())

raw_df.dropna(subset=['RainToday','RainTomorrow'], inplace=True)

#corealtion and causation

sns.set_style("darkgrid")
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["figure.facecolor"] = '#00000000'

fig =   px.histogram(raw_df,
                    x='Location',
                    title='Location vs Rainydays',
                    color='RainToday' 
                    )

#fig.show()

#print(raw_df.Location.nunique())

fig2 = px.histogram(
                    raw_df,
                    x='Temp3pm',
                    title='temp at 3am vs Rain tomorrow',
                    color='RainTomorrow'

                    )

#fig2.show()


#print(raw_df.Temp3pm.nunique())

fig3 = px.histogram(raw_df,
                    x='RainToday',
                    color="RainTomorrow")

#fig3.show()

fig4 = px.scatter(raw_df,
                  x='MinTemp',
                  y='MaxTemp',
                  color="RainToday"
                )

#fig4.show()



#How to use sample 
#here dataset is not large so we need not use sample  

use_sample = False
sample_fraction= 0.1

if use_sample:
    raw_df = raw_df.sample(frac=sample_fraction).copy()



#Train test split (we wont be using this as there is the consideration of date and time)

#train_val_df, test_df = train_test_split(raw_df,test_size=0.2, random_state=42)
#train_df, val_df = train_test_split(train_val_df,test_size=0.25, random_state=42)


#print('train_df:', train_df.shape)
#print('test_df:', test_df.shape)
#print('val_df:', val_df.shape)

plt.title('NO. of Rowa per year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year);


#plt.show()


year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

#print('train_df: \n', train_df)
#print('test_df:\n', test_df)
#print('val_df:', val_df)

input_cols = list(train_df.columns)[1:-1]
target_cols = 'RainTomorrow'

#print(input_cols)

train_inputs = train_df[input_cols].copy()
train_targets =train_df[target_cols].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_cols].copy()

test_inputs = test_df[input_cols].copy()
test_targets =test_df[target_cols].copy()


#seperating numeric and catogorical columns

numeric_cols = train_inputs.select_dtypes(include= np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

#print(numeric_cols)
#print(categorical_cols)


#Performing imputation i.e. filling NaN or missing values

imputer = SimpleImputer(strategy='mean')

#print(train_inputs[numeric_cols].isna().sum())

imputer.fit(raw_df[numeric_cols])

#print(list(imputer.statistics_))

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

#print(train_inputs[numeric_cols].isna().sum())


#scaling i.e. scaling the values from 0 to 1

scaler = MinMaxScaler()

scaler.fit(raw_df[numeric_cols]) # looks for minimum and maximum from the dataset

#print(list(scaler.data_min_))
#print(list(scaler.data_max_))

train_inputs[numeric_cols] =  scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] =  scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] =  scaler.transform(test_inputs[numeric_cols])

#print(train_inputs[numeric_cols].describe())

#Encoding categorical data like location etc

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(raw_df[categorical_cols])
#print(encoder.categories_)
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#print(encoded_cols)

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

pd.set_option('display.max_columns', None)

#print(test_inputs)



#saving datasets in parquet format
train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')

pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')



#training logistic regression model

model = LogisticRegression(solver='liblinear') #initiating

model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)#trainingdauhdudhauhdhvhjghghjghjvbjhgjhtvmct    gcf

#print(model.coef_.tolist())

#print(model.intercept_)

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

train_preds = model.predict(X_train)

#print(train_preds)

train_probs = model.predict_proba(X_train)# shows probability

#print(pd.DataFrame(train_probs))

#print(model.classes_)

accuracy_score(train_targets, train_preds)# accuracy

#confusion matrix

confusion_matrix(train_targets, train_preds, normalize='true')

def predict_plot(inputs, targets, name=''):
    preds = model.predict(inputs)

    accuracy = accuracy_score(targets, preds)

    print(f'Accuracy:{accuracy*100:.2f}%' )
    
    cf = confusion_matrix(targets, preds, normalize='true')

    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.title('{} confusion Matrix'.format(name))
    plt.show()

val_preds = predict_plot(X_val,val_targets,'validation' )