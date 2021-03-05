import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys
import os

cwd = os.getcwd()
filepath = cwd +"/"+"sample-data.csv"

ds = pd.read_csv(filepath)
item_lst = []
for item_det in ds['description']:
	item_lst.append(item_det)
newLst = []
for item in item_lst:	
	newLst.append(item.split(' - ')[0])
ds['itemName'] = newLst

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['description'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in ds.iterrows():
	similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
	similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

	results[row['id']] = similar_items[1:]
	
def item(id):
	return ds.loc[ds['id'] == id]['itemName'].tolist()[0]

def recommend(item_id, num):
	print("***********************************")
	print("Recommending " + str(num) + " products similar to " + item(item_id) + "as below::")
	print("-----------------------------------")
	recs = results[item_id][:num]
	for rec in recs:
		print(item(rec[1]) + " (score:" + str(rec[0]) + ")")

if __name__ == "__main__":
	item_id = input("Enter the itemID ranging between 1 to 500:")
	num = (input("Enter the number of recommendations needed to be displayed:"))
	recommend(int(item_id), int(num))
