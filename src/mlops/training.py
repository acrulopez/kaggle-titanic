import statsmodels.api as sm
from sklearn import svm
import numpy as np
import sklearn.ensemble as ske

from constants import LOGIT_FORMULA, SVC_RF_FORMULA
from utils import set_seed, split_x_y


@set_seed
def train_logit(df, formula=LOGIT_FORMULA):
    """Train a dataset with a support vector classifier

    Args:
        df (pandas.DataFrame): dataframe containing the dataset
        formula (str): formula to apply to the split x-y

    Returns:
        model: the model
    """

    # create a regression friendly dataframe using patsy's dmatrices function
    y, x = split_x_y(df, formula)

    # instantiate our model
    model = sm.Logit(y, x)

    # fit our model to the training data
    res = model.fit()

    return res


@set_seed
def train_svc(df, kernel="poly", features=[2, 3], gamma=3, formula=SVC_RF_FORMULA):
    """Train a dataset with a logistic regression algorithm

    Args:
        df (pandas.DataFrame): dataframe containing the dataset
        kernel (str, optional): kernel to apply to the svc. Defaults to "poly".
        features (list, optional): features to use on the model. Defaults to [2, 3].
        gamma (int, optional): gamma to apply to the svc. Defaults to 3.
        formula (str): formula to apply to the split x-y

    Returns:
        model: the model
    """

    y, x = split_x_y(df, formula)
    X = np.asarray(x)
    y = np.asarray(y)
    # Get the features to train the svc
    X = X[:, features]
    # y needs to be 1 dimenstional so we flatten. it comes out of dmatirces with a shape.
    y = y.flatten()

    n_sample = len(X)
    order = np.random.permutation(n_sample)
    X = X[order]
    y = y[order].astype(np.float)

    clf = svm.SVC(kernel=kernel, gamma=gamma).fit(X, y)
    return clf


@set_seed
def train_random_forest(df, formula=SVC_RF_FORMULA):
    """Train a dataset with the random forest algorithm

    Args:
        df (pandas.DataFrame): dataframe containing the dataset
        formula (str): formula to apply to the split x-y

    Returns:
        model: the model
    """
    y, x = split_x_y(df, formula)
    # RandomForestClassifier expects a 1 demensional NumPy array, so we convert
    y = np.asarray(y).ravel()
    # instantiate and fit our model
    model = ske.RandomForestClassifier(n_estimators=100).fit(x, y)
    return model
