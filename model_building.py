import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from loggers import *
from plotters import *


"""Helper functions for model building and logging"""
def auc_scorer() -> make_scorer:
    score = make_scorer(roc_auc_score,multi_class='ovo',response_method='predict_proba')
    return score

auc_scorer = auc_scorer()

def class_names_aux(labels: pd.DataFrame) -> list:
    max_labels = max(labels)
    class_names = [str(int(i)) for i in range(1, int(max_labels)+1)]
    return class_names


"""Main parameters for model building and evaluation"""

criterions = ['gini', 'entropy']

feature_names = ['Min_WSA_Z1','Avg_WSA_Z1','Max_WSA_Z1','Min_WSA_Z2','Avg_WSA_Z2','Max_WSA_Z2','Min_WSA_Z4','Avg_WSA_Z4','Max_WSA_Z4','Min_WSA_Z5','Avg_WSA_Z5','Max_WSA_Z5','Min_WSA_Z7','Avg_WSA_Z7','Max_WSA_Z7','Min_WSA_Z8','Avg_WSA_Z8','Max_WSA_Z8','Min_WSA_Z9','Avg_WSA_Z9','Max_WSA_Z9','Min_WSA_Z10','Avg_WSA_Z10','Max_WSA_Z10','Min_WSA_Z11','Avg_WSA_Z11','Max_WSA_Z11','Min_WSA_Z12','Avg_WSA_Z12','Max_WSA_Z12','Z1_Z2_Difference','Z4_Z5_Difference','Z7_Z9_Difference','Z10_Z12_Difference','Min_Temp_Z1','Avg_Temp_Z1','Max_Temp_Z1','Min_Temp_Z2','Avg_Temp_Z2','Max_Temp_Z2','Min_Temp_Z4','Avg_Temp_Z4','Max_Temp_Z4','Min_Temp_Z5','Avg_Temp_Z5','Max_Temp_Z5','Min_Temp_Z7','Avg_Temp_Z7','Max_Temp_Z7','Min_Temp_Z8','Avg_Temp_Z8','Max_Temp_Z8','Min_Temp_Z9','Avg_Temp_Z9','Max_Temp_Z9','Min_Temp_Z10','Avg_Temp_Z10','Max_Temp_Z10','Min_Temp_Z11','Avg_Temp_Z11','Max_Temp_Z11','Min_Temp_Z12','Avg_Temp_Z12','Max_Temp_Z12']

"""
Demo values
decision_tree_params = {'min_samples_leaf': range(1, 6, 2),
                        'min_samples_split': range(2, 9),
                        'criterion': criterions}


random_forest_params = {'n_estimators': [100],
                        'criterion': criterions,
                        'min_samples_split': range(2, 3),
                        'min_samples_leaf': range(1, 2)}
"""


"""Experiments Values"""

decision_tree_params = {'min_samples_leaf': range(1, 11),
                        'min_samples_split': [0.05,0.1,2,3,4,5,6,7,8,9],
                        'criterion': criterions}


random_forest_params = {'n_estimators': [100,300,500,700],
                        'criterion': criterions,
                        'min_samples_split': [0.05,0.1,1,2,5,7,9,10],
                        'min_samples_leaf': [0.05,1,3,5,7,9],
                        'max_features': ['sqrt', None]}



scoring = {'f1_macro': make_scorer(f1_score,average='macro'),
           'balanced_accuracy': make_scorer(balanced_accuracy_score),
           'precision': make_scorer(precision_score,average='macro',zero_division=0),
           'recall_macro': make_scorer(recall_score,average='macro',zero_division=0),
           'roc_auc': auc_scorer}



def load_data(input_path: Path,year: str,region:str,dataset: str)->tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from CSV file and return features and labels
    @:param input_path: Path to the input CSV file
    @:param year: Year of the data
    @:param region: Region of the data
    @:param dataset: Dataset type (ds1, ds2, or ds3)
    @:return: Tuple of features (pd.DataFrame) and labels (pd.DataFrame)"""

    data = pd.read_csv(input_path/f'{year}_{region}_mom.csv')

    if dataset == 'ds1':
        features,labels = data.iloc[:, :30], data.iloc[:,[-1]]
    elif dataset == 'ds2':
        features,labels = data.iloc[:, :34], data.iloc[:,[-1]]
    else:
        features,labels = data.iloc[:, :-1], data.iloc[:,[-1]]


    labels = np.ravel(labels)
    return features, labels



def build_decision_tree(features: pd.DataFrame, labels: pd.DataFrame, depth: int, splits=10, test_size=0.2)->tuple[DecisionTreeClassifier, list, list]:

    """Build a decision tree classifier with given features and labels
    @:param features: DataFrame containing the features for training
    @:param labels: DataFrame containing the labels for training
    @:param depth: Maximum depth of the decision tree
    @:param splits: Number of splits for cross-validation (default is 10)
    @:param test_size: Proportion of the dataset to include in the test split (default is 0.2)
    @:return: Tuple containing the trained DecisionTreeClassifier, statistics, class names, and depth of the tree"""

    clf = DecisionTreeClassifier(max_depth=depth,random_state=42)
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=test_size, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42,stratify=labels)

    grid = GridSearchCV(clf, decision_tree_params, scoring=scoring, cv=sss, n_jobs=-1, refit= 'f1_macro',verbose=1)

    grid.fit(x_train, y_train)
    clf = DecisionTreeClassifier(**grid.best_params_,max_depth=depth)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)

    stats, test_report = log_stats(grid,y_test,y_pred,y_score,'dt')

    print_stats(*stats[3:],test_report)

    return clf,stats,class_names_aux(labels)



def build_random_forest(features: pd.DataFrame,labels: pd.DataFrame,splits=10,test_size=0.2):

    """Function to build a Random Forest classifier with given features and labels
    @:param features: DataFrame containing the features for training
    @:param labels: DataFrame containing the labels for training
    @:param splits: Number of splits for cross-validation (default is 10)
    @:param test_size: Proportion of the dataset to include in the test split (default is 0.2)
    @:return: Tuple containing the trained RandomForestClassifier and statistics
    """

    clf = RandomForestClassifier(random_state=42)
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=test_size, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42,stratify=labels)
    grid = GridSearchCV(clf, random_forest_params, scoring=scoring, cv=sss, n_jobs=-1, refit='f1_macro',verbose=1)
    grid.fit(x_train, y_train)
    clf = RandomForestClassifier(**grid.best_params_,random_state=42)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)

    stats, test_report = log_stats(grid,y_test,y_pred,y_score,'rf')
    print_stats(*stats[4:],test_report)
    return clf,stats



def results_saving_rf(stats_list: list,year: str, region: str, output_path: Path, clf: RandomForestClassifier, mode:str, dataset: str)-> None:
    """Function to save the results of Random Forest model training and evaluation
    @:param stats_list: List containing statistics of the model
    @:param year: Year of the data
    @:param region: Region of the data
    @:param output_path: Path to save the results
    @:param clf: Trained RandomForestClassifier
    @:param dataset: Dataset type (ds1, ds2, or ds3)
    @:return: None"""

    stats_arr = np.array(stats_list).reshape(1,-1)
    stats_df = pd.DataFrame(stats_arr, columns=['Criterion','Min_samples_leaf','Min_samples_split','N_Estimators','Accuracy','Precision','Recall','F1','AUC','Accuracy','Precision','Recall','F1','AUC'])

    stats_df.to_excel(output_path/f'{year}_{region}_classification_performance_rf_{mode}.xlsx')
    plot_forest_feature_importances(clf, feature_names, year, region, dataset, output_path)



def log_decision_tree(model: DecisionTreeClassifier, class_names: list, depth: int, region: str, year: str, mode: str, output_path: Path) -> list:
    """Function to extract rules from a trained Decision Tree model
    @:param model: Trained DecisionTreeClassifier
    @:param feature_names: List of feature names used in the model
    @:param class_names: List of class names used in the model
    @:return: List of rules extracted from the model"""

    rules = get_rules(model, feature_names, class_names)
    rule_list = []
    for r in rules:
        #print(r)
        rule_separator = r.split('(proba')
        stability_period_separator = r.split('SP')
        samples_rule_separator = r.split('| based on ')
        rule = rule_separator[0]
        stability_period = stability_period_separator[1][2]
        samples = samples_rule_separator[1][0] + samples_rule_separator[1][1] + samples_rule_separator[1][2]
        rule_info = [int(stability_period), rule, int(samples)]
        rule_list.append(rule_info)
    rule_list.append(['---', '---', '---'])

    plot_decision_tree(model,year,region,str(depth),feature_names,class_names, mode, output_path)

    return rule_list


def results_saving_dt(rule_list: list, stats_list: list, year: str, region: str, mode:str, output_path: Path) -> None:
    """Function to save the results of Decision Tree model training and evaluation
    @:param rule_list: List containing rules extracted from the Decision Tree model
    @:param stats_list: List containing statistics of the model
    @:param year: Year of the data
    @:param region: Region of the data
    @:param output_path: Path to save the results
    @:return: None"""

    stats_arr = np.array(stats_list)
    stats_df = pd.DataFrame(stats_arr, columns=['Criterion','Min_samples_leaf','Min_samples_split','Accuracy','Precision','Recall','F1','AUC','Accuracy','Precision','Recall','F1','AUC'])

    stats_df.to_excel(output_path/f'{year}_{region}_classification_performance_dt_{mode}.xlsx')

    rule_list = np.array(rule_list)
    rule_df = pd.DataFrame(rule_list, columns=['Stability Period (SP)', 'Rule', 'Samples'])

    rule_df.to_excel(output_path/f'{year}_{region}_decision_tree_rules_{mode}.xlsx')
