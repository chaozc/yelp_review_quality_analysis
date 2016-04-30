from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import cross_validation
from sklearn import svm, linear_model
from sklearn.metrics import classification_report
import json
# from nltk import word_tokenize          
# from nltk.stem import WordNetLemmatizer 
# import nltk
# from nltk.corpus import wordnet
# from nltk.tokenize import RegexpTokenizer

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    else:
        return None

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		"""
		tagged = nltk.pos_tag(word_tokenize(doc))
		pos_tags = [(t, get_wordnet_pos(pos)) for (t, pos) in tagged]
		return [self.wnl.lemmatize(t, pos) if pos != None else t for (t, pos) in pos_tags]
		"""
		tokenizer = RegexpTokenizer(r'\w+')
		return [self.wnl.lemmatize(t) for t in tokenizer.tokenize(doc)]

def raw_feature_extraction(infname, cntt):
	inf = open(infname, 'r')
	texts = []
	labels = []
	cntt1 = 0
	cntt2 = 0
	for line in inf:
		js = json.loads(line)
		lb = 1 if (js["votes"]["funny"]+js["votes"]["cool"]+js["votes"]["useful"] > 0) else 0
		if lb == 1:
			cntt1 += 1
		elif cntt2 < cntt1:
			cntt2 += 1
		else:
			lb = -1
		if lb > -1:
			labels.append(lb)
			texts.append(js["text"])
		cntt -= 1
		if cntt == 0:
			break
	inf.close()
	return texts, labels

def user_feature_extraction(infname, cntt):
	inf = open(infname, 'r')
	user_info = []
	labels = []
	cntt1 = 0
	cntt2 = 0
	for line in inf:
		user_info_row = []
		js = json.loads(line)
		lb = 1 if (js["votes"]["funny"]+js["votes"]["cool"]+js["votes"]["useful"] > 0) else 0
		if lb == 1:
			cntt1 += 1
		elif cntt2 < cntt1:
			cntt2 += 1
		else:
			lb = -1
		if lb > -1:
			user_info_row.append(js["user"]["votes"]["funny"]+js["user"]["votes"]["useful"]+js["user"]["votes"]["cool"])
			user_info_row.append(js["user"]["review_count"])
			user_info_row.append(js["user"]["fans"])
			user_info_row.append(js["user"]["average_stars"])
			compliments = 0
			for key, value in js["user"]["compliments"].iteritems():
				compliments = compliments + value
			user_info_row.append(compliments)
			elite = 0;
			if(bool(js["user"]["elite"])):
				elite = 1
			user_info_row.append(elite)
			user_info.append(user_info_row)
			labels.append(lb)
		cntt -= 1
		if cntt == 0:
			break
	inf.close()
	return user_info, labels

def data_split(X, Y, test_size=0.2):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=0)
	return X_train, X_test, y_train, y_test

def tfidf_extraction(texts, vectorizer=None, max_features=2000, ngram_range=(1,1)):
	if (vectorizer == None):
		vectorizer = TFIDF(tokenizer=LemmaTokenizer(), binary="True", strip_accents="unicode", ngram_range=ngram_range, analyzer='word', stop_words='english', max_features=max_features)
		tfidf = vectorizer.fit_transform(texts)
	else:
		tfidf = vectorizer.transform(texts)
	return tfidf, vectorizer

def feature_select_chi2(X, Y, ch2=None, k=1000):
	if (ch2 == None):
		ch2 = SelectKBest(chi2, k=k)
	X = ch2.fit_transform(X, Y)
	return X, ch2

def my_SVM(X_train, X_test, y_train, y_test):
	print "	-- SVM training ..."
	clf = svm.SVC()
	clf.fit(X_train, y_train)

	print "	-- SVM predicting ..."
	y_pre = clf.predict(X_test)
	return y_pre, classification_report(y_test, y_pre)

def my_Logistic(X_train, X_test, y_train, y_test, C=1):
	print " -- Logistic Training ..."
	logreg = linear_model.LogisticRegression(C=C)
	logreg.fit(X_train, y_train)
	print " -- Logistic Predicting ..."
	#y_pre = logreg.predict(X_test)
	y_pre = logreg.decision_function(X_test)
	y_pre = [1 if it > -0.5 else 0 for it in y_pre]
	return y_pre,classification_report(y_test, y_pre)

if __name__ == "__main__":
	print "Extracting raw features ..."
	user_info, labels = user_feature_extraction("../data/review_user.json", 500000)

	print "Spliting data ..."
	print len(user_info), len(labels)
	user_info_train, user_info_test, y_train, y_test = data_split(user_info, labels)

	print "Training and Testing with logistic regression"
	y_pre,report = my_Logistic(user_info_train, user_info_test, y_train, y_test, C=1)
	print(report)






