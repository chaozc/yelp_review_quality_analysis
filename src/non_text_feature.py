from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import cross_validation
from sklearn import svm, linear_model
from sklearn.metrics import classification_report
import json

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import numpy as np
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

def ada_boost(X_train, X_test, y_train, y_test, C=1):
	# print " -- ada_boost Training ..."
	# logreg = linear_model.LogisticRegression(C=C)
	# logreg.fit(X_train, y_train)
	# print " -- ada_boost Predicting ..."
	# y_pre = logreg.predict(X_test)
	# return y_pre, classification_report(y_test, y_pre)
	# Construct dataset
	# X1, y1 = make_gaussian_quantiles(cov=2.,
	#                                  n_samples=200, n_features=2,
	#                                  n_classes=2, random_state=1)
	# X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
	#                                  n_samples=300, n_features=2,
	#                                  n_classes=2, random_state=1)
	
	# count1=0
	# count2 = 0
	print(type(y_test))
	# [ X1[count++], y1[count++] = X_train(i), y_train(i) for i in range(len(y_train)) if y_train(i)==1 ]
	# for i in range(len(y_train.count)):
	# 	if X_train[i]==1:
	# 		X1[count1]  = X_train[i]
	# 		y1[count1] = X_train[i] 
	# 	else:
	# 		X2[count2++]  = X_train[i]
	X1 = []
	X2 = []
	y1 = []
	y2 = []
	for x, y in zip(X_train, y_train):
		if y==1:
			y1.append(y)
			X1.append(x)
		else:
			y2.append(y)
			X2.append(x)

	# count = 0
	# [ X2[count++], y2[count++]  = X_train(i), y_train(i) for i in range(len(y_train))if y_train(i)==0  ]
	print(y1.count(1))

	print(y2.count(0))
	X1 =np.asarray(X1)
	X2 =np.asarray(X2)
	y1 = np.asarray(y1)
	y2 = np.asarray(y2)
	# y = np.asarray(y)
	X = np.concatenate((X1, X2))
	y = np.concatenate((y1, y2))

	# Create and fit an AdaBoosted decision tree
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
	                         algorithm="SAMME",
	                         n_estimators=200)

	bdt.fit(X, y)

	# plot_colors = "br"
	# plot_step = 0.02
	# class_names = "AB"

	# plt.figure(figsize=(10, 5))

	# Plot the decision boundaries
	# plt.subplot(121)
	# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	# xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
	#                      np.arange(y_min, y_max, plot_step))

	# Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
	# Z = Z.reshape(xx.shape)
	# cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
	# plt.axis("tight")

	# Plot the training points
	# for i, n, c in zip(range(2), class_names, plot_colors):
	#     idx = np.where(y == i)
	#     plt.scatter(X[idx, 0], X[idx, 1],
	#                 c=c, cmap=plt.cm.Paired,
	#                 label="Class %s" % n)
	# plt.xlim(x_min, x_max)
	# plt.ylim(y_min, y_max)
	# plt.legend(loc='upper right')
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.title('Decision Boundary')

	# Plot the two-class decision scores
	twoclass_output = bdt.decision_function(X)

	print(type(twoclass_output))

	# import IPython
	# IPython.embed()

	# return twoclass_output, classification_report(y, twoclass_output)
	# y_pre = bdt.decision_function(X_test)
	y_pre = bdt.predict(X_test)
	# y_pre = [1 if it > 0 else 0 for it in y_pre]

	return y_pre,classification_report(y_test, y_pre)
	# return y_pre,classification_report(y_test, y_pre)
	# plot_range = (twoclass_output.min(), twoclass_output.max())
	# plt.subplot(122)
	# for i, n, c in zip(range(2), class_names, plot_colors):
	    # plt.hist(twoclass_output[y == i],
	    #          bins=10,
	    #          range=plot_range,
	    #          facecolor=c,
	    #          label='Class %s' % n,
	    #          alpha=.5)
	# x1, x2, y1, y2 = plt.axis()
	# plt.axis((x1, x2, y1, y2 * 1.2))
	# plt.legend(loc='upper right')
	# plt.ylabel('Samples')
	# plt.xlabel('Score')
	# plt.title('Decision Scores')

	# plt.tight_layout()
	# plt.subplots_adjust(wspace=0.35)
	# plt.show()

if __name__ == "__main__":
	print "Extracting raw features ..."
	user_info, labels = user_feature_extraction("../data/review_user.json", 500000)

	print "Spliting data ..."
	print len(user_info), len(labels)
	user_info_train, user_info_test, y_train, y_test = data_split(user_info, labels)
	# print(type(y_test))
	# import IPython
	# IPython.embed()
	print "Training and Testing with logistic regression"
	y_pre,report = ada_boost(user_info_train, user_info_test, y_train, y_test, C=1)
	print(report)


