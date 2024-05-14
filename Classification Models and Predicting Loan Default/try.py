# import packages
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

'''loading'''

# Both features and target have already been scaled: mean = 0; SD = 1
df = pd.read_excel('LC_data_for_PPA2.xlsx')

# Convert 'home_ownership' to 'owns_home_with_mortgage' and 'owns_home_outright'
df['owns_home_with_mortgage'] = (df['home_ownership'] == 'MORTGAGE').astype(int)
df['owns_home_outright'] = (df['home_ownership'] == 'OWN').astype(int)

# Convert 'grade' to 'LC_grade_X'
grades = ['A', 'B', 'C', 'D']
for grade in grades:
    df[f'LC_grade_{grade}'] = (df['grade'] == grade).astype(int)

# If all 'LC_grade_X' are 0, then the loan grade must be 'F'
df['LC_grade_F'] = ((df['LC_grade_A'] == 0) & (df['LC_grade_B'] == 0) &
                    (df['LC_grade_C'] == 0) & (df['LC_grade_D'] == 0)).astype(int)

# Convert 'emp_length' to 'years_employed'
df['emp_length'] = df['emp_length'].replace('< 1 year', '0.5 years')
df['emp_length'] = df['emp_length'].replace('10+ years', '10 years')
df['emp_length'] = df['emp_length'].replace(np.nan, '0 years')
df['years_employed'] = df['emp_length'].str.extract('(\d+)').astype(int)

# Convert 'term' to 'loan_term_months'
df['loan_term_months'] = df['term'].str.extract('(\d+)').astype(int)
df.to_excel('new.xlsx')

'''Part A'''

columns = ['default', 'annual_inc', 'dti', 'loan_amnt', 'fico_range_high', 'delinq_2yrs']
df_a = pd.DataFrame()

for i in columns:
    df_a[i] = df[i]

df_a = df_a.dropna()
X = df_a.drop(columns='default')
y = df_a['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

freq = y_train.value_counts()  # count frequency of different classes in training swet
freq / sum(freq) * 100  # get percentage of above

# Create an ionstance of logisticregression named lgstc_reg

lgstc_reg = LogisticRegression(penalty="none", solver="newton-cg")

# Fit logististic regression to training set

lgstc_reg.fit(X_train, y_train)  # fit training data on logistic regression

print(lgstc_reg.intercept_, lgstc_reg.coef_)  # get the coefficients of each features

# y_train_pred, and y_test_pred are the predicted probabilities for the training set
# validation set and test set using the fitted logistic regression model

y_train_pred = lgstc_reg.predict_proba(X_train)
y_test_pred = lgstc_reg.predict_proba(X_test)

# Calculate maximum likelihood for training set, validation set, and test set

mle_vector_train = np.log(np.where(y_train == 1, y_train_pred[:, 1], y_train_pred[:, 0]))
mle_vector_test = np.log(np.where(y_test == 1, y_test_pred[:, 1], y_test_pred[:, 0]))

# Calculate cost functions from maximum likelihoods

cost_function_training = np.negative(np.sum(mle_vector_train) / len(y_train))
cost_function_test = np.negative(np.sum(mle_vector_test) / len(y_test))

print('cost function training set =', cost_function_training)
print('cost function test set =', cost_function_test)

THRESHOLD = [.75, .80, .85]
# Create dataframe to store resultd
results = pd.DataFrame(
    columns=["THRESHOLD", "accuracy", "true pos rate", "true neg rate", "false pos rate", "precision",
             "f-score"])  # df to store results

# Create threshold row
results['THRESHOLD'] = THRESHOLD

j = 0

# Iterate over the 3 thresholds

for i in THRESHOLD:
    # lgstc_reg.fit(X_train, y_train)

    # If prob for test set > threshold predict 1
    preds = np.where(lgstc_reg.predict_proba(X_test)[:, 1] > i, 1, 0)

    # create confusion matrix
    cm = (confusion_matrix(y_test, preds, labels=[1, 0], sample_weight=None) / len(
        y_test)) * 100  # confusion matrix (in percentage)

    print('Confusion matrix for threshold =', i)
    print(cm)
    print(' ')

    TP = cm[0][0]  # True Positives
    FN = cm[0][1]  # False Positives
    FP = cm[1][0]  # True Negatives
    TN = cm[1][1]  # False Negatives

    results.iloc[j, 1] = accuracy_score(y_test, preds)
    results.iloc[j, 2] = recall_score(y_test, preds)
    results.iloc[j, 3] = TN / (FP + TN)  # True negative rate
    results.iloc[j, 4] = FP / (FP + TN)  # False positive rate
    results.iloc[j, 5] = precision_score(y_test, preds)
    results.iloc[j, 6] = f1_score(y_test, preds)

    j += 1

print('ALL METRICS')
print(results.T)

# Calculate the receiver operating curve and the AUC measure

lr_prob = lgstc_reg.predict_proba(X_test)
lr_prob = lr_prob[:, 1]
ns_prob = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_prob)
lr_auc = roc_auc_score(y_test, lr_prob)
print("AUC random predictions =", ns_auc)
print("AUC predictions from logistic regression model =", lr_auc)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_prob)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Predction')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('AUC.png')
plt.show()

'''Part.B'''

import pandas as pd
import statsmodels.api as sm


# Define the features and target variable
features = ['loan_amnt', 'loan_term_months', 'int_rate', 'installment',
            'LC_grade_A', 'LC_grade_B', 'LC_grade_C', 'LC_grade_D',
            'years_employed', 'owns_home_with_mortgage', 'owns_home_outright',
            'delinq_2yrs', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
            'tot_cur_bal', 'annual_inc', 'dti', 'fico_range_high']
target = 'default'
total = features + [target]
df_b = df[total]
df_b = df_b.dropna()

# Create a stepwise function
def stepwise_logit(X, y, initial_features=[], threshold_in=0.05, threshold_out=0.1):
    included = list(initial_features)
    while True:
        changed = False

        # Add variables that improve the model
        excluded = [var for var in X.columns if var not in included]
        new_pval = pd.Series(index=excluded)
        for new_var in excluded:
            model = sm.Logit(y, sm.add_constant(X[included + [new_var]])).fit(disp=False)
            new_pval[new_var] = model.pvalues[new_var]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_var = new_pval.idxmin()
            included.append(best_var)
            changed = True

        # Remove variables that worsen the model
        model = sm.Logit(y, sm.add_constant(X[included])).fit(disp=False)
        pvalues = model.pvalues.iloc[1:]  # Exclude the intercept
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_var = pvalues.idxmax()
            included.remove(worst_var)
            changed = True

        if not changed:
            break

    return included


# Perform stepwise variable selection
selected_features = stepwise_logit(df_b[features], df_b[target])

# Fit the logistic regression model with the selected features
X = df_b[selected_features]
X = sm.add_constant(X)
y = df_b[target]
model = sm.Logit(y, X).fit()

# Print the summary of the model
print(model.summary())

'''Part.C'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, export_text
from IPython.display import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, average_precision_score

features = ['loan_amnt', 'loan_term_months', 'installment',
            'years_employed', 'owns_home_with_mortgage', 'owns_home_outright',
            'delinq_2yrs', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
            'tot_cur_bal', 'annual_inc', 'dti', 'fico_range_high']
target = 'default'
total = features + [target]
df_c = df[total]
df_c = df_c.dropna()

X = df_c.drop(columns='default')
y = df_c['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier(criterion='entropy',max_depth=4,min_samples_split=1000,min_samples_leaf=200,random_state=0)
clf = clf.fit(X_train,y_train)
fig, ax = plt.subplots(figsize=(40, 30))
plot_tree(clf, filled=True, feature_names=X_train.columns, proportion=True)
plt.savefig('tree.png')
plt.show()

y_train_pred = clf.predict_proba(X_train)
y_test_pred=clf.predict_proba(X_test)

print (y_train_pred)

# Calculate maximum likelihood for training set and test set

mle_vector_train = np.log(np.where(y_train == 1, y_train_pred[:,1], y_train_pred[:,0]))
mle_vector_test = np.log(np.where(y_test == 1, y_test_pred[:,1], y_test_pred[:,0]))

# Calculate cost functions from maximum likelihoods

cost_function_training=np.negative(np.sum(mle_vector_train)/len(y_train))
cost_function_test=np.negative(np.sum(mle_vector_test)/len(y_test))

print (y_train_pred)
print('cost function training set =', cost_function_training)
print('cost function test set =', cost_function_test)

THRESHOLD = [.75, .80, .85]
results = pd.DataFrame(
    columns=["THRESHOLD", "accuracy", "true pos rate", "true neg rate", "false pos rate", "precision",
             "f-score"])  # df to store results
results['THRESHOLD'] = THRESHOLD  # threshold column
n_test = len(y_test)
Q = clf.predict_proba(X_test)[:, 1]

j = 0
for i in THRESHOLD:  # iterate over each threshold
    # fit data to model
    preds = np.where(Q > i, 1, 0)  # if prob > threshold, predict 1

    cm = (confusion_matrix(y_test, preds, labels=[1, 0], sample_weight=None) / n_test) * 100
    # confusion matrix (in percentage)

    print('Confusion matrix for threshold =', i)
    print(cm)
    print(' ')

    TP = cm[0][0]  # True Positives
    FN = cm[0][1]  # False Positives
    FP = cm[1][0]  # True Negatives
    TN = cm[1][1]  # False Negatives

    results.iloc[j, 1] = accuracy_score(y_test, preds)
    results.iloc[j, 2] = recall_score(y_test, preds)
    results.iloc[j, 3] = TN / (FP + TN)  # True negative rate
    results.iloc[j, 4] = FP / (FP + TN)  # False positive rate
    results.iloc[j, 5] = precision_score(y_test, preds)
    results.iloc[j, 6] = f1_score(y_test, preds)

    j += 1

print('ALL METRICS')
print(results.T.to_string(header=False))


# Compute the ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, Q)
roc_auc = auc(fpr,tpr)
plt.figure(figsize=(8,6))      # format the plot size
lw = 1.5
plt.plot(fpr, tpr, color='darkorange', marker='.',
         lw=lw, label='Decision Tree (AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
         label='Random Prediction (AUC = 0.5)' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

'''Part.D'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset

# Define the features and target variable
features = ['loan_amnt', 'loan_term_months', 'int_rate', 'installment',
            'LC_grade_A', 'LC_grade_B', 'LC_grade_C', 'LC_grade_D',
            'years_employed', 'owns_home_with_mortgage', 'owns_home_outright',
            'delinq_2yrs', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
            'tot_cur_bal', 'annual_inc', 'dti', 'fico_range_high']
target = 'default'
total = features + [target]
df_d = df[total]
df_d = df_d.dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_d[features], df_d[target], test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Generate ROC curve and calculate AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

'''Part.E'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


# Define the features and target variable
features = ['loan_amnt', 'loan_term_months', 'int_rate', 'installment',
            'LC_grade_A', 'LC_grade_B', 'LC_grade_C', 'LC_grade_D',
            'years_employed', 'owns_home_with_mortgage', 'owns_home_outright',
            'delinq_2yrs', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
            'tot_cur_bal', 'annual_inc', 'dti', 'fico_range_high']
target = 'default'

total = features + [target]
df_e = df[total]
df_e = df_e.dropna()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_e[features], df_e[target], test_size=0.2, random_state=42)

# Create and train the Gradient Boosting Machine classifier
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbm_model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Generate ROC curve and calculate AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()




# Get the feature importance scores
feature_importance = gbm_model.feature_importances_

# Create a DataFrame to display the feature importance
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the relative feature importance
print("Relative Feature Importance:")
print(feature_importanc
_df)