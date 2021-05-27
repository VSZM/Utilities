# **VSZM's Data Science Cheat Sheet**

## **Data Science Projects**

#### How to structure Projects 

https://github.com/pyscaffold/pyscaffoldext-dsproject


### Common imports

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline for static images in notebooks
from IPython.core.pylabtools import figsize


sns.set(font_scale=1.5)
figsize(10, 10)
```



### Kaggle data import 

https://www.kaggle.com/general/74235


## **Exploratory Data Analysis EDA**



#### Display dataframe head rows

```python
df.head()
```

#### dataframe statistics

```python
df.describe()
```

#### Check if column in Dataframe is monotonic increasing

```python
df_mp.index.is_monotonic_increasing
```



#### Identify highly correlated columns for dropping

```python
corr_matrix = df.corr().abs()
upper_triangle_corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
redundant_columns = [column for column in upper_triangle_corr_matrix.columns if any(upper_triangle_corr_matrix[column] > 0.98)]
df = df.drop(redundant_columns, axis=1) # drop
```



### Missing value handling

#### 1. Drop

```python
df.drop('STALK_ROOT', axis = 1, inplace = True)
```

#### 2. Impute 

```
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```


#### 3. Impute and mark imputation

```python
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
for col in cols_with_missing:
    imputed_X_train[col + '_was_missing'] = X_train[col].isnull()
    imputed_X_valid[col + '_was_missing'] = X_valid[col].isnull()
```


### String categorical variable handling

#### String column identification

```
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
```

#### 1. Drop

```python
df.drop('STALK_ROOT', axis = 1, inplace = True)
```

#### 2. Label encoding (transform string to number)

```python
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
```

#### 3. Onehot encoding (Creat flag column for each string)

```python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
```


#### One hot encoding with pandas

```python
telco_df = telco_df.join(pd.get_dummies(telco_df['region'], prefix='region'), how='inner')
```

### Normalization


#### boxcox box cox transformation

https://towardsdatascience.com/box-cox-transformation-explained-51d745e34203


## **Randomization**

### Generating Random numbers

#### Exponential distribution

!(https://www.itl.nist.gov/div898/handbook/eda/section3/gif/expcdf.gif)


#### generate numbers evenly split range, interval

```python
np.linspace(0, 10, 100)
```

### Data

#### Randomize shuffle rows

```python
df = df.sample(frac=1).reset_index(drop=True)
```

## **Number crunching**

### Manipulating numbers

#### Round to nearest 100 

```python
np.floor(waiting_times / 100) * 100
```

## **Utilities**


### Jupyter

#### git filter to strip notebook output

```python
pip install nbstripout
https://github.com/kynan/nbstripout
nbstripout --install
```

#### jupyter runtime metrics extension

```python
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

### Displaying numbers, printing

#### numpy print normally, no scientific notation

```python
numpy.set_printoptions(precision=4, suppress=True, threshold=10000) # or sys.maxsize
```

#### pandas print normally, no scientific notation

```python
pd.set_option('display.float_format', lambda x: '%.4f' % x)
```



#### pandas config display columns rows

```python
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
```

### IO Input Output 

#### read csv

```python
df = pd.read_csv('asd.csv', delimiter=';')
```

#### Export Dataframe

##### Export to excel


```python
df.to_excel('results.xlsx', sheet_name='Sheet', index=False)
```
[Pretty formatting excel](https://www.pbpython.com/improve-pandas-excel-output.html)

##### Multiple Dataframes to excel export

```python
writer = pd.ExcelWriter('pandas_multiple.xlsx', engine='xlsxwriter')

df1.to_excel(writer, sheet_name='Sheet1')
df2.to_excel(writer, sheet_name='Sheet2')
df3.to_excel(writer, sheet_name='Sheet3')

writer.save()
```

##### Append to existing excel

```python
excelBook = load_workbook(filename)
with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
    # Save your file workbook as base
    writer.book = excelBook
    writer.sheets = dict((ws.title, ws) for ws in excelBook.worksheets)

    # Now here add your new sheets
    result.to_excel(writer,'saw', index = False)

    # Save the file
    writer.save() 
```


## **Working with Dataframes**

### Creating Dataframes

#### Create Dataframe from lists

```python
df = pd.DataFrame(zip(['a', 'b'], [1, 2]), columns = ['strings', 'numbers'])
```

### Iterating Dataframes 

#### Dataframe to list

```python
df.values.tolist()
```

#### iterate rows like each row is a dict

```python
for row in df.to_dict(orient='records'):
    pass
```    


#### dataframe to list of dicts

```python
df.to_dict('records')
```

### Filtering Dataframe


#### Index dataframe df multiple columns

```python
df[['X','Y']]
```

#### filter dataframe by column value 

```python
df[df['charge_time'] < 4.0]
df = df.query('state != "live"') #modern
```

### Modify Dataframe 

#### append, extend df 

```python
df = df_a.append(df_b, ignore_index=True)
```

#### Drop, remove dataframe df column

```python
df.drop('STALK_ROOT', axis = 1, inplace = True)
```

#### Add column to Dataframe

```python
df = df.assign(outcome=(df['state'] == 'successful').astype(int))
df['new col'] = some_list
```

#### Insert column at specified location

```python
df.insert(loc=idx, column='Column Name', value=values_list)
```

#### update modify cell value in dataframe by index

```python
df.at[idx, 'Columnt'] = 'New Value'
```

#### Dataframe drop duplicate rows by certain column

```python
df.drop_duplicates(subset='Document', keep=False, inplace=True)
```

#### Dataframe set new index

```python
df.set_index("Document", inplace=True)
```


#### Dataframe string strip whitespace

```python
df['Column'] = df['Column'].str.strip()
```

#### Static column value replacements, mapping

```python
column_replacements = {
                        "IS_EDIBLE":     {"EDIBLE": np.uint8(1), "POISONOUS": np.uint8(0)},
                        "HAS_BRUISES": {"BRUISES": True, "NO": False},
                        "RING_NUMBER": {"NONE": np.uint8(0), "ONE": np.uint8(1), "TWO": np.uint8(2)}
                      }

df.replace(column_replacements, inplace=True, )
df.head()
```

### Misc Dataframe

#### Dataframe distinct value count

```python
df['my_column'].nunique()
```

#### Sort DataFrame by column values

```
df.sort_values('column', axis = 1, ascending = False)
```



#### Fully print dataframe

```python
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
```


## **Visualization**

### Examples

#### Visualize null columns

```python
plt.figure(figsize = (20,20))
sns.heatmap(df.isnull(), cbar = False, cmap = 'viridis')
```



#### visualize linearity

```python
sns.pairplot(df, kind='reg')
```

#### Group by plot count sns seaborn

```python
groups = df.groupby(['Category']).count().reset_index()
groups.rename(columns={'id':'Cardinality'}, inplace=True)

plt.figure(figsize=(20,20))
sns.set(font_scale=2)
fig = sns.barplot(x = 'Category', y = 'Cardinality', data = groups, palette="deep")

for index, row in groups.iterrows():
    fig.text(index, row['Cardinality'], row['Cardinality'], color='black', ha="center")

fig.set_xticklabels(fig.get_xticklabels(), rotation=-45)
fig
```


#### Heatmap, correlations between features

```python
sns.heatmap(df[df.columns[:10]].corr(),annot=True)# first 10
sns.heatmap(df[df.columns[10:20].insert(0,df.columns[0])].corr(),annot=True) # second 10 with the first (important) column
```

#### visualize predictions vs actual values

```python
plt.scatter(y_test,predictions) # 1: pred vs actual


dfplot = pd.DataFrame({'charge_time': X_test[:,0], 'real': y_test, 'pred': y_pred})
dfplot = dfplot.melt(id_vars=['charge_time'], value_vars=['real', 'pred'], var_name='battery_times')
sns.scatterplot(x="charge_time", y="value", hue="battery_times", data=dfplot) # pred vs actual vs X
```

### Visualization Misc

#### Plotly export to HTML

```python
fig.write_html("path/to/file.html")
```


#### Set default figure size in jupyter notebook

```python
from IPython.core.pylabtools import figsize
figsize(10, 10)
```

#### Bigger text font size in seaborn sns

```python
sns.set(font_scale=1.5)
```

#### rotate x labels seaborn sns

```python
fig = sns.someplot()
fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
```

#### set figure size

```python
plt.figure(figsize = (20,20))
```




## **Modeling**

### **Genetic Programming**

https://gplearn.readthedocs.io/en/stable/

### **Classification**

#### Decision tree classifier

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
```

#### Keras Classification Example 1 simple

```python
import keras


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

predictions = model.predict(test_images)

np.argmax(predictions[0])

#### Keras Example 2 thorough, visualization of epoch
input_shape = (64, 64, 1)
num_classes = 2
batch_size=128
epochs=50


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                activation='relu',
                input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

model.summary()


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('val_acc'))
        
history = AccuracyHistory()


model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[history])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(range(1,len(history.acc) + 1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
```

### **Regression**


#### Support Vector Regression 

good for nonlinear models

```python
from sklearn.svm import SVR


SVR(kernel='rbf', C=8, gamma=0.1, epsilon=.1)
```

#### Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

X = [[0.44, 0.68], [0.99, 0.23]]
vector = [109.85, 155.72]
predict= [[0.49, 0.18]]
#Edit: added second square bracket above to fix the ValueError problem

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X)
predict_ = poly.fit_transform(predict)

model = linear_model.LinearRegression()
model.fit(X_, vector)
print model.predict(predict_)
```

#### Spline regression 

better than poly, less overfitting, works only with 1 d feature data

```python
from scipy.interpolate import UnivariateSpline


model = UnivariateSpline(X_train, y_train, s=1)

predictions = model(X_valid)
```


### **Clustering**

#### Clustering KMeans


```python
from sklearn.cluster import KMeans

ds_clustered = ds[['X','Y']]
kmeans = KMeans(n_clusters=7, random_state=0)
kmeans.fit(ds_clustered)

ds_clustered['Group'] = kmeans.labels_


plt.figure(figsize = (20,20))
sns.stripplot(data=ds_clustered, x='X', y='Y', hue='Group', size=15)
```

### **Time Series**


#### Smoothing averaging
https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788

#### Time series analysis
https://towardsdatascience.com/common-time-series-data-analysis-methods-and-forecasting-models-in-python-f0565b68a3d8
https://www.datacamp.com/community/tutorials/time-series-analysis-tutorial

#### Time Series feature selection and correlation analysis

https://erdem.pl/2020/06/finding-correlations-in-time-series-data

#### Stationarity and differencing

https://otexts.com/fpp2/stationarity.html

#### Time series white noise detection

https://machinelearningmastery.com/white-noise-time-series-python/

#### Understanding autocorrelation

https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm

#### Higher order visualization

https://www.jstatsoft.org/article/view/v025c01/v25c01.pdf

#### Forecasting

https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/

### Modeling Misc

#### train test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 101)
```

#### Classification metrics report

```python
from sklearn.metrics import classification_report, mean_absolute_error

print(classification_report(y_test,predictions))
print("-"*80)
print(mean_absolute_error(y_test, predictions))
```

#### visualize decision tree

```python
import graphviz 

dot_data = dt.export_graphviz(dt, out_file=None, 
                         feature_names=df.columns[1:],  
                         class_names=['Edible', 'Poisonous'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("mushroom_dt_viz")
graph
```

#### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, predictions, labels = [enum_item.value for enum_item in FragmentType.__members__.values()])

# Normalizing values for heatmap
conf_matrix = conf_matrix/conf_matrix.sum(axis=1, keepdims=True)

fig = sns.heatmap(conf_matrix, annot=True, fmt='.2%', linewidths=.5, cmap='coolwarm', cbar=False, xticklabels=labels, yticklabels=labels)
fig.set_ylabel('Actual')
fig.set_xlabel('Predicted')
```
