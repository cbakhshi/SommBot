# Import Dependencies

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import Data

red_wine = pd.read_csv('red_wine_data_index.csv')
red_wine.head()


# Assign X (data) and y (target)

# X = red_wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
X = red_wine[['residual sugar', 'pH', 'alcohol']].values
y = red_wine["quality"].values.reshape(-1, 1)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler model and fit it to the training data

X_scaler = StandardScaler().fit(X_train.reshape(-1, 1))

# Transform the training and testing data using the X_scaler and y_scaler models

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Loop through different k values to see which has the highest accuracy
# Note: We only use odd numbers because we don't want any ties
train_scores = []
test_scores = []
for k in range(1, 30, 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_score = knn.score(X_train_scaled, y_train)
    test_score = knn.score(X_test_scaled, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
    
    
plt.plot(range(1, 30, 1), train_scores, marker='o')
plt.plot(range(1, 30, 1), test_scores, marker="x")
plt.xlabel("k neighbors")
plt.ylabel("Testing accuracy Score")
plt.show()

# Note that k: 13 provides the best accuracy where the classifier starts to stablize
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
print('k=13 Test Acc: %.3f' % knn.score(X_test, y_test))

new_red_wine_data = [[2.64,3.65,13.9]]
predicted_class = knn.predict(new_red_wine_data)
print(predicted_class)

# Dependencies
import tweepy
import time
import json
import random
import requests as req
import datetime
#import pandas as pd

"""
file = "../credentials.csv"
keys_df = pd.read_csv(file,index_col=False)

# Twitter credentials

auth = tweepy.OAuthHandler(keys_df['consumer_key'][0], keys_df['consumer_secret'][0])
auth.set_access_token(keys_df['access_token'][0], keys_df['access_token_secret'][0])
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
"""

consumer_key = "MUI6gEX6XkyDZYkXNmqdAiUK1"
consumer_secret = "L54Wm2TbzXeSE24HL4eDrk12OYQEDbDa2kdL4dWrg7FgiCvGxr"
access_token = "987706832792772610-v1z4f2Xlhxk9FoFxAWA83KB8VaypNAq"
access_token_secret = "DzgllGPQKlVBcmm0gHzaEiNoV7X8IMJAQaEFF9CCb6gTJ"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())



# Target Term
target_term = "@Vino_Diezel"


# Search for most recent tweet directed to the account
public_tweets = api.search(target_term, count=1, result_type="recent")


for tweet in public_tweets['statuses']:
	
	"""
	#Print tweet in JSON
	print(
	json.dumps(
		tweet,
		sort_keys=True,
		indent=4,
		separators=(
			',',
			': ')))
	url = "http://lcboapi.com/products?"
	wine = tweet['text']
	params = {"q":wine}
	twitNm = tweet['in_reply_to_screen_name']
	twitId = tweet['in_reply_to_user_id']
	
	# Perform the API call to get the wine data
	lookup_response = req.get(url, params=params)
	
	wine_category = lookup_response['result'][0]['secondary_category']
	wine_RS = lookup_response['result'][0]['sugar_in_grams_per_liter']
	wine_ABV = lookup_response['result'][0]['alcohol_content']
	wine_Media = lookup_response['result'][0]['image_thumb_url']
	wine_pairing = lookup_response['result'][0]['serving_suggestion']
	wine_Score = ""
	
	
	#Do something to retrieve score

	# Reply with the score
	api.update_with_media(wine_Media,status = "@" + twitNm + " the score for your wine is:   " + wine_Score \
			+ ".  Wine pairing suggestion:  " + wine_pairing, in_reply_to_id = twitId)
			
	"""
	
	#tweet:  ['Red', residual sugar, pH, alcohol] or Name of Wine?
	twitNm = tweet['user']['screen_name']
	twitId = tweet['user']['id']
	wine = tweet['text']
	wine_split = wine.split(',')
	wine_list = [x.strip() for x in wine_split]
	
	#type of wine
	wine_type = wine_list[0].lower()
	
	#inputs to wine models
	wine_inputs = wine_list[1:]
	
	#Red/White wine 
	#if wine_type == 'red':
	predicted_class = knn.predict(wine_inputs)
	wine_score = predicted_class
	#print(wine_score)
	#elif wine_type=='white':
		#Use the white wine model
	#else:
		#wine_score = 'Does not compute'
		
	# Reply with the score
	api.update_status("@" + twitNm + " The score for your wine is:   " + str(wine_score) \
			 ,in_reply_to_id = twitId)
