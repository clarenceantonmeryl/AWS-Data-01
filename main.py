# Imports

# Common
import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Econding
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


df = load()

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


def feature_encode(names):
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


def min_max_scaler():
    """
    :param df: Pandas DataFrame
    :return: Pandas DataFrame
    """
    scaled_array = MinMaxScaler().fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_array, columns=FEATURE_NAMES + [CLASS_NAME])
    return df_scaled


def standard_scaler():
    """
    :param df: Pandas DataFrame
    :return: Pandas DataFrame
    """
    scaled_array = StandardScaler().fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_array, columns=FEATURE_NAMES + [CLASS_NAME])
    return df_scaled


def robust_scaler():
    """
    :param df: Pandas DataFrame
    :return: Pandas DataFrame
    """
    scaled_array = RobustScaler().fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_array, columns=FEATURE_NAMES + [CLASS_NAME])
    return df_scaled


# 5: Data Balancing
def split():
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

# Encoding
maps = feature_encode(ENCODE_NAMES)

# Inputation
df = simple_impute(df, 'median')
# df = kNN_impute(df, 3) | > 2 min!

# Scaling
df = min_max_scaler()
# df = standard_scaler()
# df = robust_scaler()


# Splitting
X_train, X_test, y_train, y_test = split()


# Balancing
# X_resampled, y_resampled = random_oversampler(X_train, y_train)
# X_resampled, y_resampled = random_undersampler(X_train, y_train)
# X_resampled, y_resampled = near_miss(X_train, y_train, 1)
# X_resampled, y_resampled = near_miss(X_train, y_train, 2)
# X_resampled, y_resampled = near_miss(X_train, y_train, 3)
X_resampled, y_resampled = smote(X_train, y_train, 5)

# Tree Model
clf = decision_tree(X_resampled, y_resampled)

print(f"The accuracy of the classifier on the test set is {get_accuracy(clf, X_test, y_test) * 100}%")
