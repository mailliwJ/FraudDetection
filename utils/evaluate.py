# ----------------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
from sklearn.compose import  ColumnTransformer

from imblearn.pipeline import Pipeline as imbPipe

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def test_evaluate_pipeline(pipe, algorithm_name='', sampling_techniques=(), train_set=None, test_set=None, target=''):

    # Prepare title
    if not sampling_techniques:
        title = f'{algorithm_name.title()}'
    elif len(sampling_techniques) == 1:
        title = f'{algorithm_name.title()} with {sampling_techniques[0]}'
    elif len(sampling_techniques) == 2:
        title = f'{algorithm_name.title()} with {sampling_techniques[0]} and {sampling_techniques[1]}'
    elif len(sampling_techniques) == 3:
        title = f'{algorithm_name.title()} with {sampling_techniques[0]}, {sampling_techniques[1]} and {sampling_techniques[2]}'
    
    print('*' * len(title))
    print(title)
    print('*' * len(title))
    print()

    # Cross Validate pipeline on training set
    cv_scores = cross_validate(pipe, X=train_set.drop(target, axis=1), y=train_set[target], scoring=['recall','precision','average_precision','f1','f1_weighted'], cv=5)

    print('Cross validation on training set')
    print('-' * 32)
    print('Results:')
    for k, i in cv_scores.items():
        print(f'{k}: {round(np.mean(i), 2)}')
    print()

    # Fit model to training set and predict on test set
    pipe.fit(X=train_set.drop(target, axis=1), y=train_set[target])
    y_test = test_set[target]
    y_preds = pipe.predict(X=test_set.drop(target, axis=1))

    # Generate classification report for evaluation of model on test set
    print('Model Evaluation using Test Set')
    print('-' * 31)
    print('Classification Report:')
    print(classification_report(y_test, y_preds))
    print()

    # Generate confusion matrix for evaluation of model on test set
    print('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

    return None

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def cross_validate_imbpipes(X, y, algorithms={}, pipe_steps=[], preprocessor='v1', roc=False):

    if preprocessor == 'v1':
        preprocessor = ColumnTransformer([
            ('drop_cols', 'drop', ['Time','Amount'])],
            remainder='passthrough')
    else:
        preprocessor = preprocessor

    if preprocessor is not None:
        pipe_steps.insert(0, ('preprocessor', preprocessor))
            
    models = []
    fit_time = []
    recall = []
    precision = []
    pr_auc = []
    # roc_auc = []

    for name, alg in algorithms.items():
        pipe_steps.append(('classifier', alg))

        pipe = imbPipe(steps=pipe_steps)
        
        print(f'Cross validating {name}...')
        cv_scores = cross_validate(pipe, X=X, y=y, scoring=['recall','precision','f1','average_precision','roc_auc'], cv=5)

        models.append(name)
        fit_time.append(round(np.mean(cv_scores['fit_time']),2))
        recall.append(round(np.mean(cv_scores['test_recall']),2))
        precision.append(round(np.mean(cv_scores['test_precision']),2))
        pr_auc.append(round(np.mean(cv_scores['test_average_precision']),2))
        
        # if roc:
        #     roc_auc.append(round(np.mean(cv_scores['test_roc_auc']),2))

        results_df = pd.DataFrame({
            'Model': models,
            'Fit Time': fit_time,
            'Mean Recall': recall,
            'Mean Precision': precision,
            'PR-AUC': pr_auc,
            # 'ROC-AUC': roc_auc
            }).set_index('Model')
        
        return results_df
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def cm_evaluation(X_train, y_train, X_test, y_test, algorithms, pipelines, preprocessor='v1', cols=2):
    """
    Generates confusion matrices for classification algorithms

    Parameters:
        X_train, y_train, X_test, y_test: 
        Train/test features and target

        algorithms:
        Dictionary of algorithms, e.g.
        {'Logistic Regression': LogisticRegression(), ...}

        pipelines:
        Dictionary of pipeline configurations. Each key is a
        pipeline name and its values a list of pipeline steps
        (a list of tuples).

        preprocessor:
        Either 'v1' (to use default ColumnTransformer that drops
        ['Time, 'Amount']) or a custom preprocessr object

        cols:
        Nummber of columns to display in the subplot grid
    """
    
    num_algs = len(algorithms)
    rows = ceil(num_algs / cols)
    figsize = (cols * 6.5, rows * 5)
    
    y_true = y_test

    for pipe_name, pipe_steps in pipelines.items():
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if num_algs ==1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (alg_name, alg) in enumerate(algorithms.items()):
            if preprocessor == 'v1':
                preproc = ColumnTransformer(
                    transformers=[('drop_cols', 'drop', ['Time', 'Amount'])],
                    remainder='passthrough'
                )
            else:
                preproc = preprocessor

            steps = pipe_steps.copy()

            if preproc:
                steps.insert(0, ('preprocessor', preproc))

            steps.append(('classifier', alg))

            pipe = imbPipe(steps=steps)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])
            disp.plot(cmap='Purples', values_format='d', ax=axes[i])
            axes[i].set_title(alg_name)
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Confusion Matrices for the Pipeline: {pipe_name}')
        
        plt.tight_layout()
        plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
