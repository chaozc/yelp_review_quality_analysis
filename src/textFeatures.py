from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import cross_validation
from sklearn import svm, linear_model
from sklearn.metrics import classification_report
import json
def raw_feature_extraction(infname, cntt):
	inf = open(infname, 'r')
	texts = []
	labels = []
	for line in inf:
		js = json.loads(line)
		texts.append(js["text"])
		lb = 1 if (js["votes"]["funny"]+js["votes"]["cool"]+js["votes"]["useful"] > 0) else 0
		labels.append(lb)
		cntt -= 1
		if cntt == 0:
			break
	inf.close()
	return texts, labels

def data_split(X, Y):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
	return X_train, X_test, y_train, y_test


def tfidf_extraction(texts, vectorizer=None, max_features=2000):
	if (vectorizer == None):
		vectorizer = TFIDF(binary="True", strip_accents="unicode", ngram_range=(1,1), analyzer='word', stop_words='english', max_features=max_features)
		tfidf = vectorizer.fit_transform(texts)
	else:
		tfidf = vectorizer.transform(texts)
	return tfidf, vectorizer

def feature_select_chi2(X, Y, ch2=None):
	if (ch2 == None):
		ch2 = SelectKBest(chi2, k=1000)
	X = ch2.fit_transform(X, Y)
	return X, chi2

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
	y_pre = logreg.predict(X_test)
	return y_pre, classification_report(y_test, y_pre)

if __name__ == "__main__":
	#tfidf_extraction("../data/review_user.json")
	print "Extracting raw features ..."
	texts, labels = raw_feature_extraction("../data/review_user.json", 50000)

	print "Spliting data ..."
	texts_train, texts_test, y_train, y_test = data_split(texts, labels)

	print "Building tfidf model ..."
	X_train, vectorizer = tfidf_extraction(texts=texts_train)

	print "Fitting testset to tfidf model ..."
	X_test, vectorizer = tfidf_extraction(texts_test, vectorizer)

	print "Trying SVM ..."
	y_pre, res_svm = svm(X_train, X_test, y_train, y_test)
	print res_svm