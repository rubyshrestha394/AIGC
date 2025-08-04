from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_svm(X_train, y_train):
    svm = SVC(probability=True, random_state=42)
    param_grid = {'C': [1], 'kernel': ['linear'], 'class_weight': ['balanced']}
    grid = GridSearchCV(svm, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_rf(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {'n_estimators': [100], 'max_depth': [None]}
    grid = GridSearchCV(rf, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_xgb(X_train, y_train):
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    param_grid = {'n_estimators': [100], 'max_depth': [3], 'scale_pos_weight': [1.0]}
    grid = GridSearchCV(xgb, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
