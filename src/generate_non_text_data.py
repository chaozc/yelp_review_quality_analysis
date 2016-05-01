from textFeatures import *
from non_text_feature import *
import json

inf = open('../data/review_user.json', 'r')
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
        texts.append(line)


inf.close()


texts_train, texts_test, y_train, y_test = data_split(texts, labels, test_size=0.2)
ouf = open('../data/train.json', 'w')
for line in texts_train[:120000]:
    ouf.write(line)
ouf.close()
ouf = open('../data/test.json', 'w')
for line in texts_test[:30000]:
    ouf.write(line)
ouf.close()
print "finished"


def my_random_forest(X_train, X_test, y_train, y_test, n_estimators):
    print " -- RandomForestClassifier training ..."
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf = clf.fit(X_train, y_train)

    print " -- RandomForestClassifier testing ..."
    y_pre = clf.predict(X_test)
    return y_pre, classification_report(y_test, y_pre)

def raw_feature(oufname):
    inf = open(oufname, 'r')
    features = []
    lbs = []
    for line in inf:
        js = json.loads(line)
        lb = 1 if (js["votes"]["funny"]+js["votes"]["cool"]+js["votes"]["useful"] > 0) else 0
        user_info_row = []
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
        features.append(user_info_row)
        lbs.append(lb)
    return features, lbs

def test_ada_boost():
    X_train, Y_train = raw_feature("../data/train.json")
    X_test, Y_test = raw_feature("../data/test.json")
    y_pre,report = ada_boost(X_train, X_test, Y_train, Y_test, C=1)
    print(report)
    print "finished"

def test_svm(): #test non_text feature using svm
    X_train, Y_train = raw_feature("../data/train.json")
    X_test, Y_test = raw_feature("../data/test.json")
    y_pre,report = non_text_svm(X_train, X_test, Y_train, Y_test, C=1)
    print(report)
    print "finished"

if __name__ == "__main__":
    test_svm()
