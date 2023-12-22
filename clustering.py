import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import silhouette_score
import pickle

def recency_score(df):
  max = df['Recency'].max()
  return (1-(df.Recency - max))/max




df = pd.read_csv('data/Online-Retail.csv')
print(df)

df["TotalSpent"] = df['UnitPrice'] * df['Quantity']
#Adding recency
df.InvoiceDate = pd.to_datetime(df.InvoiceDate)
maxdate =max(df.InvoiceDate)
df['Recency'] = maxdate-df['InvoiceDate']

print(df.head())

df['Recency'] = df['Recency'].dt.total_seconds() / (60 * 60 * 24)  # Convert to days

# Define segmentation criteria
spending_threshold = df['TotalSpent'].mean()  # Example threshold for high spenders
recency_threshold = df['Recency'].mean()   # Example threshold for recent customers

# Segment customers
high_spender_recent = df[(df['TotalSpent'] > spending_threshold) & (df['Recency'] < recency_threshold)]
low_spender_recent = df[(df['TotalSpent'] <= spending_threshold) & (df['Recency'] < recency_threshold)]
high_spender_inactive = df[(df['TotalSpent'] > spending_threshold) & (df['Recency'] >= recency_threshold)]
low_spender_inactive = df[(df['TotalSpent'] <= spending_threshold) & (df['Recency'] >= recency_threshold)]

recency = high_spender_recent[['CustomerID','Quantity','Recency']]

recency['RecentScore'] = recency_score(recency)*100

grouped_df = recency.groupby('CustomerID').agg({
    'Quantity':'sum', 'Recency':"max", 'RecentScore': 'mean'

}).reset_index()

print(f"\n \n",grouped_df.head())

X = grouped_df.drop('CustomerID', axis=1)
ms = MinMaxScaler()

X = ms.fit_transform(X)
estimator = KMeans(n_clusters=3, n_init=10)
estimator.fit(X)
cc = estimator.cluster_centers_

print('Inertia is:', estimator.inertia_)
print(cc)
print(cc.shape)

silhouette_score = silhouette_score(X, estimator.labels_)
with open('freezed_data.pkl', 'wb') as f:
    pickle.dump({
        'centroids': cc,
        'inertia': estimator.inertia_,
        'silhoutte_score': silhouette_score
    },f)

print('Data are being saved to the pickle file.')