# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from math import sqrt, ceil
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score)


os.chdir("/Users/Rawshan/Desktop/Thesis/")


#%% read & prepare data. 

# data_all = pd.read_excel("Class_ransomware.xlsx", sheet_name = 0, header = 0, index_col = 0)
# data_all = pd.read_csv("Class.csv", header = 0, index_col = -1)
data_all = pd.read_csv("Class_new.csv", header = 0, index_col = -1)

## get features & numeric labels. 
features = data_all.reset_index(drop = True)               # remove rw names
labels = LabelEncoder().fit_transform( data_all.index )    # encode rw names to classes

## feature correlation with label. 
corr_features = pd.DataFrame(
    [(feat, *pearsonr(features[feat], labels)) for feat in features.columns], 
    columns = ["feature", "cor", "pval"])
keep = (corr_features.cor.abs() >= 0.25) & (corr_features.pval < 0.05)
corr_features = corr_features[keep]

corr_features_mat = pd.concat(
    [features, pd.Series(labels, name = "Class")], axis = 1).corr(method = "pearson")
keep2 = corr_features.feature.tolist() + ["Class"]
corr_features_mat = corr_features_mat.loc[keep2, keep2]

# plots. 
fig, ax = plt.subplots(
    num = 1, figsize = (10, 8), nrows = 1, ncols = 2, clear = True);
# feature heatmap. 
sns.heatmap(data = corr_features_mat, cmap = "jet", ax = ax[0]);
ax[0].tick_params(axis = "both", labelsize = 8);
# feature corr w/ label. 
sns.barplot(x = "cor", y = "feature", data = corr_features, ax = ax[1]);
ax[1].set_xlabel("Pearson correlation");    ax[1].set_ylabel(None);
ax[1].tick_params(axis = "y", label1On = False);

fig.suptitle("Feature correlation with multiclass label", 
             fontweight = "semibold", y = 0.96);
plt.show()


#%% build classifiers with hyperparameter tuning. 

## tuning subroutine with grid search. 
def hyperparameter_tuning(clf, params, train_X, train_Y, verbose = 1):
    grid = GridSearchCV(
        estimator = clf, param_grid = params, scoring = "accuracy", 
        cv = 3, refit = True, n_jobs = -1, verbose = verbose)
    
    grid.fit(train_X, train_Y)
    
    return grid.best_estimator_


## build tuned classifiers. 

def LogReg(train_X, train_Y, rng = None, verbose = 0):
    lr = hyperparameter_tuning(
        clf = LogisticRegression(
            multi_class = "multinomial", solver = "saga", l1_ratio = 0.7, 
            tol = 1e-4, max_iter = int(1e3), class_weight = "balanced", 
            n_jobs = -1, random_state = rng), 
        params = dict(
            penalty = ["l2", "l1", "elasticnet", "none"], 
            C = np.arange(0.7, 1, 0.1)), 
        train_X = train_X, train_Y = train_Y, verbose = verbose)
    
    return lr

def StocGradDes(train_X, train_Y, rng = None, verbose = 0):
    sgd = hyperparameter_tuning(
        clf = SGDClassifier(
            loss = "modified_huber", learning_rate = "optimal", l1_ratio = 0.2, 
            tol = 1e-4, max_iter = int(1e3), class_weight = "balanced", 
            validation_fraction = 0.1, early_stopping = True, 
            n_iter_no_change = 5, n_jobs = -1, random_state = rng), 
        params = dict(
            penalty = ["l2", "l1", "elasticnet"], 
            alpha = [1e-6, 1e-5, 1e-4]), 
        train_X = train_X, train_Y = train_Y, verbose = verbose)
    
    return sgd


def KNearestNeigh(train_X, train_Y, rng = None, verbose = 0):
    np.random.seed(rng)    
    knn = hyperparameter_tuning(
        clf = KNeighborsClassifier(
            weights = "distance", metric = "manhattan", n_jobs = -1), 
        params = dict(
            n_neighbors = [3, 5, 7], 
            leaf_size = [7, 15, 30]), 
        train_X = train_X, train_Y = train_Y, verbose = verbose)
    
    return knn

def NaiveBayes(train_X, train_Y, rng = None, verbose = 0):
    np.random.seed(rng)
    nb = hyperparameter_tuning(
        clf = GaussianNB(priors = None), 
        params = dict(var_smoothing = [1e-6, 1e-7, 1e-8]), 
        train_X = train_X, train_Y = train_Y, verbose = verbose)
    
    return nb
    
def RandForest(train_X, train_Y, rng = None, verbose = 0):
    rf = hyperparameter_tuning(
        clf = RandomForestClassifier(
            criterion = "gini", max_features = "auto", 
            class_weight = "balanced", n_jobs = -1, random_state = rng), 
        params = dict(
            n_estimators = [50, 100, 200], 
            min_samples_split = [5, 7, 9], 
            min_samples_leaf = [3, 5, 7]), 
        train_X = train_X, train_Y = train_Y, verbose = verbose)
    
    return rf

def SupVecMach(train_X, train_Y, rng = None, verbose = 0):
    svm = hyperparameter_tuning(
        clf = SVC(
            degree = 3, gamma = "scale", probability = True, tol = 1e-4, 
            class_weight = "balanced", decision_function_shape = "ovr", 
            max_iter = int(1e4), random_state = rng), 
        params = dict(
            kernel = ["linear", "rbf", "poly"], 
            C = [0.8, 1, 10]), 
        train_X = train_X, train_Y = train_Y, verbose = verbose)
    
    return svm


#%% perform k-fold cross-validation. 

pred_th = 0.5                   # use default threshold for now
rng = 86420                     # random seed for reproducibility
verbose = 1                     # prints tuning progress; use 0 to NOT print anything
avg = "weighted"                # multi-class averaging scheme

n_cls = np.unique(labels).size

## use random_state = None if you want different split every time. 
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = rng)
clf_perf = [ ]

for fold, (train_idx, test_idx) in enumerate(kfold.split(features, labels), start = 1):
    print(f"{'-'*20} Fold {fold} {'-'*20} \n")
    
    ## training-test datasets for current fold. 
    train_X, train_Y = features.iloc[train_idx].copy(), labels[train_idx].copy()
    test_X,  test_Y  = features.iloc[test_idx].copy(),  labels[test_idx].copy()
    
    ## filter feature by SD & standardize. 
    keep_ = (train_X.std().values >= 0.0)
    train_X, test_X = train_X.iloc[:, keep_].copy(), test_X.iloc[:, keep_].copy()

    zscore = StandardScaler().fit(train_X)
    train_X[:], test_X[:] = zscore.transform(train_X), zscore.transform(test_X)
    
    
    ## build classifiers. 
    lr  = LogReg(train_X, train_Y, rng = rng, verbose = verbose)
    sgd = StocGradDes(train_X, train_Y, rng = rng, verbose = verbose)
    knn = KNearestNeigh(train_X, train_Y, rng = rng, verbose = verbose)
    nb  = NaiveBayes(train_X, train_Y, rng = rng, verbose = verbose)
    rf  = RandForest(train_X, train_Y, rng = rng, verbose = verbose)
    svm = SupVecMach(train_X, train_Y, rng = rng, verbose = verbose)
    
    clf_list = {"LR": lr, "SGD": sgd, "KNN": knn, "NB": nb, "RF": rf, "SVM": svm}
    
    
    ## evaluate performance. 
    clf_perf_fold = {};    pred_Y_pr = {}
    for clf_name, clf in clf_list.items():
        clf.fit(train_X, train_Y);
        pred_Y_pr[clf_name] = np.array( clf.predict_proba(test_X) )
        
        # convert actual & predicted labels into n x c matrix, c: #classes. 
        pred_Y_2d = (pred_Y_pr[clf_name] >= pred_th).astype(int)
        test_Y_2d = label_binarize(test_Y, classes = range(n_cls))
        
        clf_perf_fold[clf_name] = {
            "ACC": accuracy_score(test_Y_2d, pred_Y_2d), 
            "Precision": precision_score(test_Y_2d, pred_Y_2d, average = avg), 
            "Recall": recall_score(test_Y_2d, pred_Y_2d, average = avg), 
            "F1-score": f1_score(test_Y_2d, pred_Y_2d, average = avg), 
            "AUROC": roc_auc_score(test_Y_2d, pred_Y_2d, average = avg), 
            "AUPRC": average_precision_score(test_Y_2d, pred_Y_2d, average = avg)
        }
    clf_perf_fold = pd.DataFrame(clf_perf_fold)
    clf_perf.append( clf_perf_fold )
    
    if verbose:
        print(f"\nFold {fold}: Accuracies of various classifiers using Th = {pred_th}: \n{clf_perf_fold.round(4)}", end = "\n\n")


## mean performance over k-folds. 
clf_perf_mean = { }
for clf_name in clf_list:
    clf_perf_all = pd.concat([perf[clf_name] for perf in clf_perf], axis = 1)
    clf_perf_mean[clf_name] = clf_perf_all.mean(axis = 1)
clf_perf_mean = pd.DataFrame(clf_perf_mean)

print(f"Mean performance over 10-fold cross-validation: \n{clf_perf_mean.round(4)}")



#%% Best model performance. 

## pick best model. 
use_metric = "AUROC"
clf_best = clf_perf_mean.loc[use_metric].idxmax()
clf_best_perf = clf_perf_mean[clf_best].copy()
print(f"\nBest model in terms of {use_metric} = '{clf_best}'\n{clf_best_perf.round(4)}")

# performance per class. 
test_Y_2d = label_binarize(test_Y, classes = range(n_cls))
pred_Y_2d = pred_Y_pr[clf_best].copy()

clf_best_perf_ovr = {};     conf_mat_ovr = []

m = ceil(sqrt(n_cls))
fig1, ax1 = plt.subplots(
    num = 2, figsize = (10, 8), nrows = m, ncols = m, clear = False)
fig2, ax2 = plt.subplots(
    num = 3, figsize = (10, 8), nrows = m, ncols = m, clear = False)
fig3, ax3 = plt.subplots(
    num = 4, figsize = (10, 8), nrows = m, ncols = m, clear = False)
for cls_ in range(test_Y_2d.shape[1]):
    # test & predicted labels for cls_. 
    test_y = test_Y_2d[:, cls_];    pred_y_pr = pred_Y_2d[:, cls_]
    pred_y = (pred_y_pr >= pred_th).astype(int)
    
    # performance metrics for cls_.
    fpr_, tpr_, th_roc_  = roc_curve(test_y, pred_y_pr)
    prec_, rcl_, th_prc_ = precision_recall_curve(test_y, pred_y_pr)
    auroc_ = roc_auc_score(test_y, pred_y_pr)
    auprc_ = average_precision_score(test_y, pred_y_pr)
    
    conf_mat = pd.DataFrame(
        confusion_matrix(test_y, pred_y, normalize = "true")[::-1, ::-1], 
        index = ["true_1", "true_0"], columns = ["pred_1", "pred_0"])
    conf_mat_ovr.append( conf_mat )
    
    clf_best_perf_ovr[cls_] = {
        "ACC": accuracy_score(test_y, pred_y), 
        "Precision": precision_score(test_y, pred_y), 
        "Recall": recall_score(test_y, pred_y), 
        "F1-score": f1_score(test_y, pred_y), 
        "AUROC": auroc_, 
        "AUPRC": auprc_ 
    }
    

    ## plots. 
    ax_ = [ax1.ravel()[cls_], ax2.ravel()[cls_], ax3.ravel()[cls_]]
    
    # confusion matrix. 
    sns.heatmap(conf_mat, cmap = "summer", annot = True, fmt = "0.2f", 
                cbar = False, ax = ax_[0]);
    ax_[0].set_title(f"$C_{{{cls_}}}$");
    
    # roc curve. 
    ax_[1].plot(fpr_, tpr_, linestyle = "-.", color = "magenta", linewidth = 1.2, 
                label = f"$C_{{{cls_}}}: AUC = {auroc_: 0.2f}$");
    ax_[1].axline([0.05, 0.05], [0.95, 0.95], color = "gray", 
                  linestyle = ":", linewidth = 0.8);
    ax_[1].axis([-0.1, 1.1, -0.1, 1.1]);    ax_[1].legend(loc = "lower right");
    
    # pr curve. 
    ax_[2].plot(rcl_, prec_, linestyle = "-.", color = "magenta", linewidth = 1.2, 
                label = f"$C_{{{cls_}}}: AveP = {auprc_: 0.2f}$");
    ax_[2].axhline(y = test_y.mean(), xmin = 0.05, xmax = 0.95, color = "gray", 
                  linestyle = ":", linewidth = 0.8);
    ax_[2].axis([-0.1, 1.1, -0.1, 1.1]);    ax_[2].legend(loc = "lower right");

fig1.suptitle("Confusion matrix per class", fontweight = "semibold", y = 0.92)

fig2.suptitle(
    f"ROC curve per class ($AUROC_{{wAvg}} = {clf_best_perf['AUROC']: 0.4f}$)", 
    fontweight = "semibold", y = 0.92);
fig2.supxlabel("TPR", fontweight = "semibold", y = 0.02);
fig2.supylabel("FPR", fontweight = "semibold", x = 0.05);

fig3.suptitle(
    f"Precision-recall curve per class ($AUPRC_{{wAvg}} = {clf_best_perf['AUPRC']: 0.4f}$)", 
    fontweight = "semibold", y = 0.92);
fig3.supxlabel("Recall", fontweight = "semibold", y = 0.02);
fig3.supylabel("Precision", fontweight = "semibold", x = 0.05);




