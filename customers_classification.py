import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
files.upload()

df = pd.read_csv('Mall_Customers.csv')
df

df.columns

x=df[['Annual Income (k$)','Spending Score (1-100)']]
from sklearn.cluster import KMeans
x

WCSS = []

for i in range(1,14):
  model_KM= KMeans(n_clusters=i)
  model_KM.fit(x)
  WCSS.append(model_KM.inertia_)

plt.plot(list(range(1,14)),WCSS,marker= 'o')
plt.xlabel("Optimal Number Of Clusters")
plt.ylabel("Within cluster Sum of Squares")

model_KM = KMeans(n_clusters=5)
model_KM.fit(x)

KMeans

plt.title('Visualize Data')
plt.xlabel("Spending score")
plt.ylabel("Annual Income")
plt.scatter(x['Annual Income (k$)'],x['Spending Score (1-100)'])
plt.show()

model_KM = KMeans(n_clusters=5)
model_KM.fit(x)
cn=model_KM.predict(x)

print(cn)

plt.title('Visualize Data')
plt.xlabel("Spending score")
plt.ylabel("Annual Income")
plt.scatter(x[cn==0]['Annual Income (k$)'],x[cn==0]['Spending Score (1-100)'],c='c')
plt.scatter(x[cn==1]['Annual Income (k$)'],x[cn==1]['Spending Score (1-100)'],c='y')
plt.scatter(x[cn==2]['Annual Income (k$)'],x[cn==2]['Spending Score (1-100)'],c='b')
plt.scatter(x[cn==3]['Annual Income (k$)'],x[cn==3]['Spending Score (1-100)'],c='g')
plt.scatter(x[cn==4]['Annual Income (k$)'],x[cn==4]['Spending Score (1-100)'],c='r')
plt.show()
