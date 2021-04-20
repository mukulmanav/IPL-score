#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv('ipl2017.csv')
#%%
df.head()
#%%
x=df.drop(columns=['total','venue','bat_team','bowl_team','batsman','bowler'] )
y=df['total']
x=x.drop(columns='date' )
# %%
#Checking null data
x.isna().sum()
y.isna().sum()
#%%
#visualising the data
x.dtypes
type(x)
x.shape

#%%
# Spliting dataset into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x.head()
x.describe()

#%%
#scaling
x.head()
x.describe()
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# %%
knn=KNeighborsRegressor(n_neighbors=2)
knn.fit(x_train,y_train)
# %%
knn.score(x_test,y_test)
#%%
value=pd.DataFrame({"mid":[1],"runs":[21],"wickets":[1],"overs":[0],"runs_last_5":[31],"wickets_last_5":[3],"striker":[11],"non-striker":[7]})
value=scaler.transform(value)
knn.predict(value)
# %%
