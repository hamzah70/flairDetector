import pickle
import pandas as pd
import dataCollector
import praw
from praw.models import MoreComments

def predictFlair(X):
	trainedModel = ''
	tfidfList = []
	with open('Trained_Model.pickle', 'rb') as f:
		trainedModel = pickle.load(f)

	with open('Tfidf_Vectorizer.pickle', 'rb') as f:
		tfidfList = pickle.load(f)

	df_predict = pd.DataFrame()
	j=0
	for i in X.columns:
		tfidf = tfidfList[j]
		df_predict_tmp = pd.DataFrame(tfidf.transform(X[i]).todense(), columns=tfidf.get_feature_names())
		df_predict = pd.concat([df_predict, df_predict_tmp], axis=1)
		j+=1

	prediction = trainedModel.predict(df_predict)
	return prediction


def recoverPost(url):
	dataDictionary = {}
	reddit = praw.Reddit(client_id='I4SSb419kCGNmA', redirect_uri='http://localhost:8080', client_secret='Y7vR4ronY3HdQO75UXokopFB-lc', user_agent='Flair Detector')
	submission = reddit.submission(url=url)
	data = []

	dataDictionary['author'] = submission.author
	dataDictionary['id'] = submission.id
	dataDictionary['num_comments'] = submission.num_comments
	dataDictionary['permalink'] = submission.permalink
	dataDictionary['score'] = submission.score
	dataDictionary['title'] = submission.title
	dataDictionary['url'] = submission.url
	dataDictionary['selftext'] = submission.selftext
	dataDictionary['created_utc'] = submission.created_utc
	comments = ''
	for top_level_comment in submission.comments:
		if isinstance(top_level_comment, MoreComments):
			continue
		comments += top_level_comment.body 
	dataDictionary['comments'] = comments

	data.append(dataDictionary)
	pandasData = pd.DataFrame(data)
	pandasData['title'] = pandasData['title'].apply(dataCollector.preprocessText)
	pandasData['selftext'] = pandasData['selftext'].apply(dataCollector.preprocessText)
	pandasData['comments'] = pandasData['comments'].apply(dataCollector.preprocessText)
	pandasData = pandasData.astype(str)

	prediction = predictFlair(pandasData)
	return prediction[0]
	

if __name__ == '__main__':
	recoverPost('asda')



