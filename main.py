# Imports

# Common
import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Encoding
from sklearn.preprocessing import LabelEncoder

# Imputing
from sklearn.impute import SimpleImputer, KNNImputer

# Scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Balancing
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss

# Miscellaneous
from sklearn.model_selection import train_test_split

# Decision Tree
from sklearn import tree

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import accuracy_score
import time

# kNN
from sklearn.neighbors import KNeighborsClassifier

# Support Vector Machines
from sklearn.svm import LinearSVC

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# ADABoost Classifier
from sklearn.ensemble import AdaBoostClassifier

# DNN
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# XG Boost
import xgboost as xgb

# Light GBM
from lightgbm import LGBMClassifier

# Preprocessing

# 1: Constants, Loading and Preliminary Observations

FEATURE_NAMES = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                 'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                 'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ', 'unknown',
                 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt', 'num_emp',
                 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship', 'own_or_self',
                 'vet_question', 'vet_benefits', 'weeks_worked', 'year']

CLASS_NAME = 'income_50k'

RANDOM_STATE = 42


def load():
    """
    :return: Pandas DataFrame
    """
    df = pd.read_csv("Data/Census-Income_train.csv")
    # Replacing ? with NaN
    for feature in FEATURE_NAMES:
        df[feature] = df[feature].replace('?', np.NaN)

    return df


# 2: Feature Encoding
ENCODE_NAMES = [
    'class_worker',
    'education',
    'hs_college',
    'marital_stat',
    'major_ind_code',
    'major_occ_code',
    'race',
    'hisp_origin',
    'sex',
    'union_member',
    'unemp_reason',
    'full_or_part_emp',
    'tax_filer_stat',
    'region_prev_res',
    'state_prev_res',
    'det_hh_fam_stat',
    'det_hh_summ',
    'mig_chg_msa',
    'mig_chg_reg',
    'mig_move_reg',
    'mig_same',
    'mig_prev_sunbelt',
    'fam_under_18',
    'country_father',
    'country_mother',
    'country_self',
    'citizenship',
    'vet_question',
    'income_50k'
]


def replace_with_nan(df_col, labels):
    """
    :param df_col: Pandas Series
    :param labels: dict feature mappings
    :return: Pandas Series
    """
    for key in labels.keys():
        if type(key) == float:
            df_col = df_col.replace(labels[key], np.NaN)
    return df_col


def feature_encode(df, names):
    """
    :param names: list of all encoded feature names
    :return: maps, which is the feature maps of each of the encoded features
    """
    le = LabelEncoder()
    maps = []
    for feature in names:
        df[feature] = le.fit_transform(df[feature])
        feature_map = dict(zip(le.classes_, le.transform(le.classes_)))
        maps.append(feature_map)
        df[feature] = replace_with_nan(df[feature], feature_map)
    return maps


# 3: Imputation

def simple_impute(df, strategy):
    '''

    :param df: Pandas DataFrame
    :param strategy: String of 'mean', 'most_frequent', 'median'
    :return: Pandas DataFrame
    '''
    imputer = SimpleImputer(strategy=strategy)

    # Fit and transform the dataset
    imputed = imputer.fit_transform(df)

    df_imputed = pd.DataFrame(imputed, columns=FEATURE_NAMES + [CLASS_NAME])
    return df_imputed


def kNN_impute(df, k):
    """

    :param df: DataFrame
    :param k: Value of k in kNN
    :return: DataFrame
    """
    # NOTE: TAKES TOO LONG (> 2 MINUTES)
    knn_imputer = KNNImputer(n_neighbors=k, weights="uniform")
    knn_imputed = knn_imputer.fit_transform(df)

    df_knn_imputed = pd.DataFrame(knn_imputed, columns=FEATURE_NAMES + [CLASS_NAME])

    return df_knn_imputed


# 4: Feature Scaling


def min_max_scaler(df):
    """
    :param df: Pandas DataFrame
    :return: Pandas DataFrame
    """
    scaled_array = MinMaxScaler().fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_array, columns=FEATURE_NAMES + [CLASS_NAME])
    return df_scaled


def standard_scaler(df):
    """
    :param df: Pandas DataFrame
    :return: Pandas DataFrame
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    # scaled_array = StandardScaler().fit_transform(df.values)
    # df_scaled = pd.DataFrame(scaled_array, columns=FEATURE_NAMES + [CLASS_NAME])
    return df_scaled


def robust_scaler(df):
    """
    :param df: Pandas DataFrame
    :return: Pandas DataFrame
    """
    scaled_array = RobustScaler().fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_array, columns=FEATURE_NAMES + [CLASS_NAME])
    return df_scaled


# 5: Data Balancing
def split(df):
    """

    :return: split of all train/test
    """
    X = df.drop(CLASS_NAME, axis=1)
    y = df[CLASS_NAME]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def random_oversampler(X_train, y_train):
    """

    :param X_train: X train dataset
    :param y_train: y train dataset
    :return: resampled X and y by random oversampler
    """
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def smote(X_train, y_train, k):
    """

    :param X_train: X train dataset
    :param y_train: y train dataset
    :param k: k value in kNN
    :return: resampled X and y by SMOTE
    """
    ros = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.25, k_neighbors=k)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def random_undersampler(X_train, y_train):
    """

    :param X_train: X train dataset
    :param y_train: y train dataset
    :return: resampled X and y by random undersampler
    """
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def near_miss(X_train, y_train, version):
    """

    :param X_train: X train dataset
    :param y_train: y train dataset
    :param version: The type of near miss algorithm used (1, 2, or 3)
    :return: resampled X and y by near miss
    """
    nm = NearMiss(version=version)
    X_resampled, y_resampled = nm.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


# Extra: 6: Correlations (Visualisations) and Feature Removal

# 7: Visualisations


# Alogrithms and Models

# 1: KNN

# 2: Decision Trees

def decision_tree(X, y):
    """

    :param X: X train
    :param y: y train
    :return: classifier
    """
    clf = tree.DecisionTreeClassifier()
    # clf = tree.DecisionTreeClassifier(criterion = 'gini')
    clf = clf.fit(X, y)

    plt.figure(figsize=(12, 12))
    tree.plot_tree(clf, fontsize=12)

    return clf


def get_accuracy(classifier, X_test, y_test):
    """

    :param classifier:
    :param X_test:
    :param y_test:
    :return: accuracy of model
    """
    y_pred = classifier.predict(X_test)
    return np.mean(y_pred == y_test)


# All Together:

def predict_incomes(imputer, impute_k, scaler, balancer, k_balance, algorithm, k_algorithm):
    """

    :param imputer: 'simple' or 'kNN'
    :param impute_k: k value of kNN imputer (put random number if not used)
    :param scaler: 'min_max', 'standard' or 'robust'
    :param balancer: 'ros', 'rus', 'nm1', 'nm2', 'nm3' or 'smote'
    :param k_balance: k value of the smote
    :param algorithm: 'tree', 'log_reg', 'kNN', 'svm', 'random_forest', 'ada', 'dnn', 'xgboost'
    :param k_algorithm: k value of the kNN algorithm
    :return:
    """
    # Load
    df = load()

    # Encoding
    maps = feature_encode(df, ENCODE_NAMES)

    # Inputation
    if imputer == 'simple':
        df_imputed = simple_impute(df, 'median')
    else:  # > 2 min!
        df_imputed = kNN_impute(df, impute_k)

    # Scaling
    if scaler == 'min_max':
        df_scaled = min_max_scaler(df_imputed)
    elif scaler == 'standard':  # 'Continuous' bug!
        df_scaled = standard_scaler(df_imputed)
    elif scaler == 'robust':
        df_scaled = robust_scaler(df_imputed)
    else:
        df_scaled = df_imputed.copy(deep=True)

    # Splitting
    X_train, X_test, y_train, y_test = split(df_scaled)

    # Balancing
    if balancer == 'ros':
        X_resampled, y_resampled = random_oversampler(X_train, y_train)
    elif balancer == 'rus':
        X_resampled, y_resampled = random_undersampler(X_train, y_train)
    elif balancer == 'nm1':
        X_resampled, y_resampled = near_miss(X_train, y_train, 1)
    elif balancer == 'nm2':
        X_resampled, y_resampled = near_miss(X_train, y_train, 2)
    elif balancer == 'nm3':
        X_resampled, y_resampled = near_miss(X_train, y_train, 3)
    elif balancer == 'smote':
        X_resampled, y_resampled = smote(X_train, y_train, k_balance)
    else:
        X_resampled = X_train
        y_resampled = y_train

    # Algorithms
    if algorithm == 'tree':
        clf = decision_tree(X_resampled, y_resampled)
        print(f"The accuracy of the decision tree on test set is {get_accuracy(clf, X_test, y_test) * 100}%")

    elif algorithm == 'log_reg':  # Convergence Warning?
        logmodel = LogisticRegression()
        logmodel.fit(X_resampled, y_resampled)
        y_pred_log = logmodel.predict(X_test)
        print(f'The accuracy of the logistic regression on test set is {accuracy_score(y_test, y_pred_log) * 100} %')

    elif algorithm == 'kNN':
        # Standardising (solution to previous bug)
        scaler = StandardScaler()
        X_resampled = scaler.fit_transform(X_resampled)
        X_test = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=k_algorithm)

        model.fit(X_resampled, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model Accuracy: {accuracy * 100}%")

    elif algorithm == 'svm':  # Future Warning
        svm_model = LinearSVC()
        svm_model.fit(X_resampled, y_resampled)
        y_pred_svm = svm_model.predict(X_test)
        print(f'The accuracy of the SVM on test set is {accuracy_score(y_test, y_pred_svm) * 100} %')

    elif algorithm == 'random_forest':
        rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=10000, n_jobs=-1, random_state=RANDOM_STATE)
        rnd_clf.fit(X_resampled, y_resampled)

        y_pred_rf = rnd_clf.predict(X_test)
        print(f'The accuracy of the Random Forest on test set is {accuracy_score(y_test, y_pred_rf) * 100} %')

    elif algorithm == 'ada':
        # learning_rate down increases accuracy at cost of time
        ada_clf = AdaBoostClassifier(
            tree.DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE)

        ada_clf.fit(X_resampled, y_resampled)

        y_pred_adab_dtree = ada_clf.predict(X_test)

        print(f'The ADABoost with decision trees accuracy is {accuracy_score(y_test, y_pred_adab_dtree) * 100} %')

    elif algorithm == 'dnn':
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        epochs = 50
        batch_size = 128
        history = model.fit(X_resampled, y_resampled, epochs=epochs, batch_size=batch_size, validation_split=0.1)

        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    elif algorithm == 'xgboost':
        model = xgb.XGBClassifier(objective='binary:logistic', random_state=RANDOM_STATE)

        model.fit(X_resampled, y_resampled)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"The accuracy of the XGBoost on test set is {accuracy * 100}")


# Running Function
start_time = time.time()

predict_incomes(
    imputer='simple',
    impute_k=0,
    scaler='min_max',
    balancer='none',
    k_balance=0,
    algorithm='xgboost',
    k_algorithm=0
)

end_time = time.time()
execution_time = end_time - start_time

print(f"The code ran in {execution_time} seconds")
