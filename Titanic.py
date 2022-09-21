import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm


# Ingestion -----------------------------------------------------------------
df = pd.read_csv("data/train.csv")

# Feature engineering -------------------------------------------------------
df = df.drop(["Ticket", "Cabin"], axis=1)
# Remove NaN values
df = df.dropna()


# model formula
# here the ~ sign is an = sign, and the features of our dataset
# are written as a formula to predict survived. The C() lets our
# regression know that those variables are categorical.
# Ref: http://patsy.readthedocs.org/en/latest/formulas.html
formula = "Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)"
# create a results dictionary to hold our regression results for easy analysis later
results = {}

# create a regression friendly dataframe using patsy's dmatrices function
y, x = dmatrices(formula, data=df, return_type="dataframe")

# instantiate our model
model = sm.Logit(y, x)

# fit our model to the training data
res = model.fit()

# save the result for outputing predictions later
results["Logit"] = [res, formula]
res.summary()


# Read the test data
test_data = pd.read_csv("data/test.csv")

# Add our independent variable to our test data. (It is usually left blank by Kaggle because it is the value you are trying to predict.)

test_data["Survived"] = 1.23


# ### Results as scored by Kaggle: RMSE = 0.77033  That result is pretty good. ECT ECT ECT

# Create an acceptable formula for our machine learning algorithms
formula_ml = "Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)"


# ### Support Vector Machine (SVM)


# create a regression friendly data frame
y, x = dmatrices(formula_ml, data=df, return_type="matrix")

# select which features we would like to analyze
# try chaning the selection here for diffrent output.
# Choose : [2,3] - pretty sweet DBs [3,1] --standard DBs [7,3] -very cool DBs,
# [3,6] -- very long complex dbs, could take over an hour to calculate!
feature_1 = 2
feature_2 = 3

X = np.asarray(x)
X = X[:, [feature_1, feature_2]]


y = np.asarray(y)
# needs to be 1 dimenstional so we flatten. it comes out of dmatirces with a shape.
y = y.flatten()

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)

X = X[order]
y = y[order].astype(np.float)

# do a cross validation
nighty_precent_of_sample = int(0.9 * n_sample)
X_train = X[:nighty_precent_of_sample]
y_train = y[:nighty_precent_of_sample]
X_test = X[nighty_precent_of_sample:]
y_test = y[nighty_precent_of_sample:]

# create a list of the types of kerneks we will use for your analysis
types_of_kernels = ["linear", "rbf", "poly"]


# Here you can output which ever result you would like by changing the Kernel and clf.predict lines
# Change kernel here to poly, rbf or linear
# adjusting the gamma level also changes the degree to which the model is fitted
clf = svm.SVC(kernel="poly", gamma=3).fit(X_train, y_train)
y, x = dmatrices(formula_ml, data=test_data, return_type="dataframe")

# Change the interger values within x.ix[:,[6,3]].dropna() explore the relationships between other
# features. the ints are column postions. ie. [6,3] 6th column and the third column are evaluated.
res_svm = clf.predict(x.ix[:, [6, 3]].dropna())

res_svm = DataFrame(res_svm, columns=["Survived"])
res_svm.to_csv(
    "data/output/svm_poly_63_g10.csv"
)  # saves the results for you, change the name as you please.


# ### Random Forest

# import the machine learning library that holds the randomforest
import sklearn.ensemble as ske

# Create the random forest model and fit the model to our training data
y, x = dmatrices(formula_ml, data=df, return_type="dataframe")
# RandomForestClassifier expects a 1 demensional NumPy array, so we convert
y = np.asarray(y).ravel()
# instantiate and fit our model
results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)

# Score the results
score = results_rf.score(x, y)
print("\n\n------------------------------------------------")
print("Mean accuracy of Random Forest Predictions on the data was: {0}".format(score))
