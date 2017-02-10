# All classifier params and defintion for train/test experimentation
# classifiers the paramters can also be modified via config files but sigh...

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier

# normalizers and mean centering or whitening
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

# These default values can be reset via python code and yaml files as f.ex test_clp.clf.pipe_line.set_params(train__class_weight=None) for settinig class_weight to None in training

classifier_dict = {'linear_svc':LinearSVC(C=10,class_weight='auto'),
					'svc':SVC(C=10, kernel='rbf', shrinking=True, probability=True,class_weight='auto'),
					'sgd_svm':SGDClassifier(loss='log', penalty="l2",class_weight='auto',n_jobs=4,n_iter=10),
					'rf_classifier':RandomForestClassifier(n_jobs=4, n_estimators=10, random_state=42),
					'rf_regressor':RandomForestRegressor(n_jobs=4, n_estimators=10, random_state=42),
					'gboost_classifier':GradientBoostingClassifier(),
					'None':None}

normalizer_dict = {'l1':Normalizer(norm='l1'),
					'l2':Normalizer(norm='l2'),
					'None':None}

scaling_dict = {'standard':StandardScaler(),'None':None}