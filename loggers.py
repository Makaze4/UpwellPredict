from sklearn.tree import _tree
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, classification_report, roc_auc_score



def prompt_user_for_input()->tuple[str, str, str]:
    """Function to prompt the user for input parameters to run the pipeline
    @:return: tuple of strings containing the year, region path and dataset"""

    #Start prompt to run the pipeline and ask for the year and region
    year = input('Enter year: ')

    # Loop until a valid year is entered
    while True:
        # Check if the year is valid with an exception to the type of the input and the range of the year
        try:
            year = int(year)
            print()
            if 2004 <= year <= 2019:
                break
            else:
                year = input('Enter a valid year between 2004 and 2019: ')
                print()
        except ValueError as e:
            year = input('Year not recognized: Enter a valid year between 2004 and 2019: ')
            print()
        except TypeError as e:
            year = input('Year not recognized: Enter a valid year between 2004 and 2019: ')
            print()

    region = input('Enter the region: N (North) or S (South): ')
    print()

    # Loop until a valid region is entered
    while True:
        if region == 'N':
            region_path = 'North'
            break
        elif region == 'S':
            region_path = 'South'
            break
        else:
            region = input('Enter a valid region: N (North) or S (South): ')
            print()

    # Loop until a valid dataset is entered
    while True:
        dataset = input('Enter the dataset to use: ds1, ds2 or ds3: ')
        print()
        if dataset in ['ds1', 'ds2', 'ds3']:
            break
        else:
            print('Invalid dataset. Please enter ds1, ds2 or ds3.')

    return str(year), region_path, dataset



def get_rules(tree, feature_names, class_names)-> list:
    """Function to extract rules from a decision tree
    @:param tree: The decision tree classifier
    @:param feature_names: List of feature names used in the decision tree
    @:param class_names: List of class names for the decision tree
    @:return: List of rules extracted from the decision tree, organized by class names"""

    rule_list = []

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 2)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 2)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "IF "

        for p in path[:-1]:
            if rule != "IF ":
                rule += " AND "
            rule += str(p)
        rule += " THEN "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            if path[-1][1] >= 10:
                rule += f"SP: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
                rule += f" | based on {path[-1][1]:,} samples"
                rules += [rule]
    rules_by_class = {class_name: [] for class_name in class_names}
    for rule in rules:
        for class_name in class_names:
            if f"SP: {class_name}" in rule:
                rules_by_class[class_name].append(rule)
                break
    # Flatten the dictionary into a list
    organized_rules = []
    for class_name in class_names:
        organized_rules.extend(rules_by_class[class_name])

    return organized_rules


def log_stats(grid,y_test,y_pred,y_score,model)-> list:
    """Function to log statistics of the model performance
    @:param grid: GridSearchCV object containing the best parameters and scores
    @:param y_test: True labels for the test set
    @:param y_pred: Predicted labels for the test set
    @:param y_score: Predicted probabilities for the test set
    @:param model: Model type ('rf' for Random Forest or 'dt' for Decision Tree)
    @:return: A list of statistics and a classification report for the test set"""

    criterion = grid.best_params_['criterion']
    min_samples_leaf = grid.best_params_['min_samples_leaf']
    min_samples_split = grid.best_params_['min_samples_split']

    val_accuracy = round(grid.cv_results_['mean_test_balanced_accuracy'][grid.best_index_],2)
    val_precision = round(grid.cv_results_['mean_test_precision'][grid.best_index_],2)
    val_recall = round(grid.cv_results_['mean_test_recall_macro'][grid.best_index_],2)
    val_f1 = round(grid.cv_results_['mean_test_f1_macro'][grid.best_index_],2)
    val_roc = round(grid.cv_results_['mean_test_roc_auc'][grid.best_index_],2)


    test_accuracy = round(balanced_accuracy_score(y_test,y_pred),2)
    test_precision = round(precision_score(y_test,y_pred,average='macro',zero_division=0),2)
    test_recall = round(recall_score(y_test,y_pred,average='macro',zero_division=0),2)
    test_f1 = round(f1_score(y_test,y_pred,average='macro',zero_division=0),2)
    test_roc = round(roc_auc_score(y_test,y_score,multi_class='ovo',average='macro'),2)
    test_report = classification_report(y_test, y_pred)

    if model == 'rf':
        n_estimators = grid.best_params_['n_estimators']
        return [criterion,int(min_samples_leaf),int(min_samples_split),int(n_estimators),val_accuracy,val_precision,val_recall,val_f1,val_roc,test_accuracy,test_precision,test_recall,test_f1,test_roc],test_report

    stats_arr = [criterion,int(min_samples_leaf),int(min_samples_split),val_accuracy,val_precision,val_recall,val_f1,val_roc,test_accuracy,test_precision,test_recall,test_f1,test_roc]

    return stats_arr,test_report



def print_stats(val_accuracy: float,val_precision: float,val_recall: float,val_f1: float,val_roc: float,test_accuracy: float,test_precision: float,test_recall: float,test_f1: float,test_roc: float,test_report: float)-> None:
    """Function to print the statistics of the model performance
    @:param val_accuracy: Validation accuracy
    @:param val_precision: Validation precision
    @:param val_recall: Validation recall
    @:param val_f1: Validation F1 score
    @:param val_roc: Validation ROC-AUC score
    @:param test_accuracy: Test accuracy
    @:param test_precision: Test precision
    @:param test_recall: Test recall
    @:param test_f1: Test F1 score
    @:param test_roc: Test ROC-AUC score
    @:param test_report: Classification report for the test set
    @:return: None"""

    print('VALIDATION')
    print('Accuracy:',val_accuracy)
    print('Precision:',val_precision)
    print('Recall:',val_recall)
    print('F1:',val_f1)
    print('ROC-AUC:',val_roc)

    print('TESTING')
    print('Balanced Accuracy:', test_accuracy)
    print('Precision:', test_precision)
    print('Recall Macro:', test_recall)
    print('F1 Macro:', test_f1)
    print('ROC-AUC:',test_roc)
    print("Test Classification Report:")
    print(test_report)

