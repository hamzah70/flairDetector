import json
import requests
import pandas as pd
import praw
from praw.models import MoreComments

# Libraries for preprocessing text data to create more meaning data
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Libraries to apply machine leaning on our data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Saving best ML model
import pickle

allModelsAccuracy = []
allModelsPredictor = []

def dataPraw():
	reddit = praw.Reddit(client_id='I4SSb419kCGNmA', redirect_uri='http://localhost:8080', client_secret='Y7vR4ronY3HdQO75UXokopFB-lc', user_agent='Flair Detector')
	subreddit = reddit.subreddit('india').hot(limit=1000)
	# subreddit = reddit.subreddit('india').rising(limit=99999)
	allData = []
	i=1
	# total eatures are 11
	for submission in subreddit:
		print(i)
		dataDictionary = {}
		dataDictionary['author'] = submission.author
		dataDictionary['id'] = submission.id
		dataDictionary['link_flair_text'] = submission.link_flair_text
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
		allData.append(dataDictionary)
		i+=1
	pandasData = pd.DataFrame(allData)
	pandasData.to_csv('subredditData.csv')



def preprocessText(text):
	if(type(text)==float):
		text=' '
	text = text.lower()
	text = text.split(" ")

	for char in text:
		if(char in string.punctuation):
			char.replace(string.punctuation, " ")

	stopword = [words for words in text if words not in stopwords.words('english')]

	lemmatized = [WordNetLemmatizer().lemmatize(words) for words in stopword]

	stemmed = " ".join([PorterStemmer().stem(words) for words in lemmatized])
	return stemmed



def loadDatafromCSV():
	data = pd.read_csv('subredditData.csv')
	data = data.astype(str)
	# print(len(data))
	data['title'] = data['title'].apply(preprocessText)
	data['selftext'] = data['selftext'].apply(preprocessText)
	data['comments'] = data['comments'].apply(preprocessText)
	X, Y = inputOutputFeature(data)
	return X, Y


def inputOutputFeature(data):
	Y = data['link_flair_text']

	data = data.drop(columns="link_flair_text")
	X = data.drop(data.columns[0], axis=1)
	print(len(X.columns))
	return X, Y

def naiveBayesModel(X_train, X_test, Y_train, Y_test):
	nb = MultinomialNB()
	nb.fit(X_train, Y_train)
	prediction = nb.predict(X_test)
	accuracy = accuracy_score(prediction, Y_test)
	print('Naive Bayes Accuracy: ', accuracy_score(prediction, Y_test))
	return accuracy, nb

def svmModel(X_train, X_test, Y_train, Y_test):
	SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	SVM.fit(X_train, Y_train)
	prediction = SVM.predict(X_test)
	accuracy = accuracy_score(prediction, Y_test)
	print("SVM Accuracy: ", accuracy_score(prediction, Y_test))
	return accuracy, SVM

def logisticRegressionModel(X_train, X_test, Y_train, Y_test):
	lr = LogisticRegression()
	lr.fit(X_train, Y_train)
	accuracy = lr.score(X_test, Y_test)
	print("Logistic Regression Accuracy: ", accuracy)
	return accuracy, lr


def knearestNeighboursModel(X_train, X_test, Y_train, Y_test):
	knr = KNeighborsClassifier()
	knr.fit(X_train, Y_train)
	prediction = knr.predict(X_test)
	accuracy = accuracy_score(prediction, Y_test)
	print("K nearest neighbours Accuracy: ", accuracy_score(prediction, Y_test))
	return accuracy, knr

def decisonTreeModel(X_train, X_test, Y_train, Y_test):
	dt = DecisionTreeClassifier()
	dt.fit(X_train, Y_train)
	prediction = dt.predict(X_test)
	accuracy = accuracy_score(prediction, Y_test)
	print("Decision Tree Regression Accuracy: ", accuracy_score(prediction, Y_test))
	return accuracy, dt

def randomForestModel(X_train, X_test, Y_train, Y_test):
	rf = RandomForestClassifier()
	rf.fit(X_train, Y_train)
	prediction = rf.predict(X_test)
	accuracy = accuracy_score(prediction, Y_test)
	print("Random Forest Regression Accuracy: ", accuracy_score(prediction, Y_test))
	return accuracy, rf

def gradientBoostingModel(X_train, X_test, Y_train, Y_test):
	gbr = GradientBoostingClassifier()
	gbr.fit(X_train, Y_train)
	prediction = gbr.predict(X_test)
	accuracy = accuracy_score(prediction, Y_test)
	print("Gradient Boosting Regression Accuracy: ", accuracy_score(prediction, Y_test))
	return accuracy, gbr



# def datacollector():
# 	allData = []
# 	# url='https://api.pushshift.io/reddit/submission/search/?subreddit=india&filter=author_flair_text,author_fullname,created_utc,id,post_hint,link_flair_text,num_comments,permalink,score,title,url,text&size=100000'
# 	# request = json.loads(requests.get(url).text)
# 	reddit = praw.Reddit(client_id='3g4ggl4suqTAIQ', redirect_uri='http://localhost:8080', client_secret='es9JYYwOFw-Fxjp04PDQ64WSwvg', user_agent='praw-test')

# 	for jsonData in request['data']:
# 		if('link_flair_text' in jsonData and jsonData['link_flair_text']!='null'):
# 			allData.append(jsonData)
# 	# 	print(type(jsonData))
# 	# 	dataDictionary = {}
# 	# 	if(jsonData!=null):
# 	# 		
# 	# print(len(allData))
# 	client = MongoClient('mongodb://localhost:27017/')
# 	db = client['REDDIT']
# 	collection = db['data']
# 	collection.remove()
# 	collection.insert_many(allData)

# def loadData():
# 	client = MongoClient('mongodb://localhost:27017/')
# 	db = client.REDDIT
# 	collection = db.data
# 	data = pd.DataFrame(list(collection.find()))
# 	print(len(data))
	# print(data)




def predictModels():
	X, Y = loadDatafromCSV()

	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)
	df_train = pd.DataFrame()
	df_test = pd.DataFrame()
	tfidfList = []
	for i in X.columns:
		
		tfidf = TfidfVectorizer(stop_words='english').fit(X_train[i])
		df_train_tmp = pd.DataFrame(tfidf.transform(X_train[i]).todense(), columns=tfidf.get_feature_names())
		df_test_tmp = pd.DataFrame(tfidf.transform(X_test[i]).todense(), columns=tfidf.get_feature_names())

		df_train = pd.concat([df_train, df_train_tmp], axis=1)
		df_test = pd.concat([df_test, df_test_tmp], axis=1)
		tfidfList.append(tfidf)

	accuracy = 0
	name = ' '

	accuracy, model = naiveBayesModel(df_train, df_test, Y_train, Y_test)
	allModelsAccuracy.append(accuracy)
	allModelsPredictor.append(model)

	accuracy, model = svmModel(df_train, df_test, Y_train, Y_test)
	allModelsAccuracy.append(accuracy)
	allModelsPredictor.append(model)

	accuracy, model = logisticRegressionModel(df_train, df_test, Y_train, Y_test)
	allModelsAccuracy.append(accuracy)
	allModelsPredictor.append(model)
	
	accuracy, model = knearestNeighboursModel(df_train, df_test, Y_train, Y_test)
	allModelsAccuracy.append(accuracy)
	allModelsPredictor.append(model)
	
	accuracy, model = decisonTreeModel(df_train, df_test, Y_train, Y_test)
	allModelsAccuracy.append(accuracy)
	allModelsPredictor.append(model)
	
	accuracy, model = randomForestModel(df_train, df_test, Y_train, Y_test)
	allModelsAccuracy.append(accuracy)
	allModelsPredictor.append(model)
	
	accuracy, model = gradientBoostingModel(df_train, df_test, Y_train, Y_test)
	allModelsAccuracy.append(accuracy)
	allModelsPredictor.append(model)

	ind = allModelsAccuracy.index(max(allModelsAccuracy))
	bestModel = allModelsPredictor[ind]
	
	with open('Trained_Model.pickle', 'wb') as f:
		pickle.dump(bestModel, f)
	with open('Tfidf_Vectorizer.pickle', 'wb') as f:
		pickle.dump(tfidfList, f)


# predictModels()

# dataPraw()
# predictModels()

# # dataPraw()
if __name__ == '__main__':
	# dataPraw()
	predictModels()




