 from sklearn.ensemble import *
 print("AdaBoostClassifier--------")
 X_train = pickle.load( open( "X_train.txt", "r" ) )
 X_test = pickle.load( open( "X_test.txt", "r" ) )
 Y_train = pickle.load( open( "Y_train.txt", "r" ) )
 Y_test = pickle.load( open( "Y_test.txt", "r" ) )
 bdt = AdaBoostClassifier(RandomForestClassifier(n_estimators = 30), algorithm="SAMME",n_estimators=300)
 bdt.fit(X_train, Y_train)
 y_pre = bdt.decision_function(X_test)


