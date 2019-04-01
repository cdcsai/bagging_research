python bagging_train_test_logreg.py --ds=digits --bagging=True --m=1 &&
python bagging_train_test_logreg.py --ds=digits --bagging=True --m=2 &&
python bagging_train_test_logreg.py --ds=digits --bagging=True --m=5 &&
python bagging_train_test_logreg.py --ds=digits --bagging=True --m=10 &&
python bagging_train_test_logreg.py --ds=digits --bagging=True --m=20 &&
python bagging_train_test_logreg.py --ds=digits --bagging=True --m=25 &&


python bagging_train_test_logreg.py --ds=digits --bagging=False --m=1 &&
python bagging_train_test_logreg.py --ds=digits --bagging=False --m=2 &&
python bagging_train_test_logreg.py --ds=digits --bagging=False --m=5 &&
python bagging_train_test_logreg.py --ds=digits --bagging=False --m=10 &&
python bagging_train_test_logreg.py --ds=digits --bagging=False --m=20 &&
python bagging_train_test_logreg.py --ds=digits --bagging=False --m=25


