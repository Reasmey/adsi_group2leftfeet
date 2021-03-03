# FUNCTIONS TO HELP LOAD AND SAVE DATA/MODELS AND OUTPUTS FASTER

# Anthony's load_sets, modified for current data
def load_sets(path='../data/processed/', val=False):
    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    import numpy as np
    import os.path

    X_train = np.load(f'{path}X_train_up.npy') if os.path.isfile(f'{path}X_train_up.npy') else None
    X_val   = np.load(f'{path}X_val_up.npy'  ) if os.path.isfile(f'{path}X_val_up.npy')   else None
    X_test  = np.load(f'{path}X_test_new.npy' ) if os.path.isfile(f'{path}X_test_new.npy')  else None
    y_train = np.load(f'{path}y_train_up.npy') if os.path.isfile(f'{path}y_train_up.npy') else None
    y_val   = np.load(f'{path}y_val_up.npy'  ) if os.path.isfile(f'{path}y_val_up.npy')   else None
    
    return X_train, y_train, X_val, y_val, X_test

# save models using one function
def save_model(model, name:str):
    """Save model outputs to models folder
    
    Parameters
    ----------
    model : model object
        model you want to save
    name : str
        the name of the model describing the type
    
    Returns
    -------
    """
    import os.path
    from joblib import dump
    
    full_name = '../models/'+name+'.joblib'
    
    if os.path.isfile(full_name) :
        print("A model by this name already exists, try again")
    else:
        dump(model, full_name)
        print('Model saved succesfully')
            


# create output file for kaggle submission
def create_output(X_preds):
    """Formats the predictions output for submission to kaggle
    
    Parameters
    ----------
    X_preds : numpy array
        The object created by .predict_proba method
    
    Returns
    -------
    pd.Dataframe 
        Id column and Target column formated for kaggle submission
    """
    import pandas as pd
    import numpy as np

    # read in Id csv
    id_col = pd.read_csv('../data/interim/test_id_col.csv')
    
    # get probabilities from predict_proba ouput
    probabilities = pd.DataFrame(X_preds[:,1], columns = ['TARGET_5Yrs'])
    
    # concat columns
    output = pd.concat([id_col,probabilities], axis=1)
    
    return output

# print model metrics WORK IN PROGRESS
def result_metrics(true_label, pred_label, pred_prob):
    """Calculates and prints performance metrics
    
    Parameters
    ----------
    true_label : numpy array
        The actual labels  
    pred_label : numpy array
        The labels predicted by .predict method
    pred_prob : numpy array
        The probabilites of class '1' created by .predict_proba method
    
    Returns
    -------
    """   
    from sklearn.metrics import roc_auc_score ,recall_score, precision_score, accuracy_score, classification_report, confusion_matrix 
    
    accuracy = accuracy_score(true_label, pred_label)
    precision=precision_score(true_label, pred_label)
    recall=recall_score(true_label, pred_label)
    roc=roc_auc_score(true_label, pred_prob[:,1])

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.2f%% " % (precision *100))
    print("Recall: %.2f%% " % (recall * 100))
    print("AUC using prediction probabilities: %.3f%% " % (roc *100))

    class_report = classification_report(true_label, pred_label)
    print(class_report)
    print('Confusion Matrix')
    print(confusion_matrix(true_label, pred_label)) 