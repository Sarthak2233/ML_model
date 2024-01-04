from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import clustering
import pandas as pd
import random
import pickle

def split(cus):
  return (cus[-1]//2)

def scale_df(df):
    data = {'Cluster': [], 'Quantity': [], 'Recency': [], 'RecentScore': [], 'TotalSpent': [], 'NumCustomers': []}

    for i , row in df.iterrows():
        cluster = row['Cluster']
        recent_score = row['RecentScore']

        if cluster == 0 and recent_score > 80:
            for j in range(100):
                data['Cluster'].append(cluster)
                data['Quantity'].append(split([283/(j+1)]))
                data['Recency'].append(random.randint(20, 29))
                data['RecentScore'].append(
                    (1 - (29 - data['Recency'][-1]) / 29)*100
                )
                data['TotalSpent'].append(random.randint(150, 268))
                data['NumCustomers'].append(split([778/(j+1)]))
        elif cluster == 1 and 38 < recent_score < 40:
            for j in range(100):
                data['Cluster'].append(cluster)
                data['Quantity'].append(split([1025/(j+1)]))
                data['Recency'].append(random.randint(20, 136))
                data['RecentScore'].append(
                    (1 - (136 - data['Recency'][-1]) / 136)*100
                )
                data['TotalSpent'].append(random.randint(1, 36))
                data['NumCustomers'].append(split([1229/(j+1)]))

        else:
            for j in range(100):
                data['Cluster'].append(cluster)
                data['Quantity'].append(split([294/(j+1)]))
                data['Recency'].append(random.randint(20, 79))
                data['RecentScore'].append(
                     (1 - (79 - data['Recency'][-1]) / 79)*100
                     )
                data['TotalSpent'].append(random.randint(22, 44))
                data['NumCustomers'].append(split([1134/(j+1)]))

    return pd.DataFrame(data)


































ms = MinMaxScaler()

grouped_df = clustering.grouped_df
X = grouped_df.drop('CustomerID', axis=1)
X = ms.fit_transform(X)
estimator = clustering.estimator
estimator.fit(X)
cc = estimator.cluster_centers_

predict = estimator.fit_predict(X)

grouped_df['Cluster'] = estimator.labels_

grouped_df = grouped_df.drop(index=grouped_df['TotalSpent'].idxmax())

cluster_centers = ms.inverse_transform(estimator.cluster_centers_)

# Create a DataFrame to show the cluster centers and the number of customers in each cluster
cluster_info = pd.DataFrame({
    'Cluster': range(3),  # Change the range depending on the number of clusters
    'Quantity': cluster_centers[:, 0],
    'Recency': cluster_centers[:, 1],
    'RecentScore': cluster_centers[:, 2],
    'TotalSpent': cluster_centers[:,3],
    'NumCustomers': grouped_df['Cluster'].value_counts().sort_index()
})
print(cluster_info.head())

generated_data = scale_df(cluster_info)
X = generated_data.drop('Quantity', axis=1)
y = generated_data.Quantity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = ms.fit_transform(X_train) #THis converts everything into numpy so
# you have o again convert it into DataFrame
X_test = ms.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse, r2)
with open('linearmodel.pkl', 'wb') as f:
    pickle.dump({
        'linearmodel': model,
        'MeanSquaredError': mse,
        'R2 Score':r2
    },f)

with open('linearmodel.pkl', 'rb') as f:
    contents = pickle.load(f)
for key, value in contents.items():
    if key == 'linearmodel':
        print(value.predict([[0,28,85,200,50]]))