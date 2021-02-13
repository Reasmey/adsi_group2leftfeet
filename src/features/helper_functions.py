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

    X_train = np.load(f'{path}X_train.npy') if os.path.isfile(f'{path}X_train.npy') else None
    X_val   = np.load(f'{path}X_val.npy'  ) if os.path.isfile(f'{path}X_val.npy')   else None
    X_test  = np.load(f'{path}X_test.npy' ) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy') if os.path.isfile(f'{path}y_train.npy') else None
    y_val   = np.load(f'{path}y_val.npy'  ) if os.path.isfile(f'{path}y_val.npy')   else None
    
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
def create_output(X_test, X_preds):
    """Formats the predictions output for submission to kaggle
    
    Parameters
    ----------
    X_test : numpy array
        The X_test data loaded using load_sets used for predictions
    X_preds : numpy array
        The object created by .predict_proba method
    
    Returns
    -------
    pd.Dataframe 
        Id column and Target column formated for kaggle submission
    """
    import pandas as pd
    import numpy as np
    
    # convert to dataframe
    df_test = pd.DataFrame(X_test)
    # extract ID col only
    id_col = df_test.iloc[:,[0]]
    # rename it
    id_col.rename(columns = {0:'Id'}, inplace = True)
    # need to change Id to int
    id_col = id_col.Id.astype(int)
    
    # get probabilities from predict_proba ouput
    probabilities = pd.DataFrame(y_test_preds_prob[:,1], columns = ['TARGET_5Yrs'])
    
    # concat columns
    output = pd.concat([id_col,probabilities], axis=1)
    
    return output

# print model metrics
def result_metrics(label, pred_probs):
    """Calculates and prints performance metrics
    
    Parameters
    ----------
    label : numpy array
        The actual labels  
    pred_probs : numpy array
        The probabilites created by .predict_proba method
    
    Returns
    -------
    """   
    from sklearn.metrics import roc_auc_score ,recall_score, precision_score, accuracy_score, classification_report
    from sklearn.metrics import plot_confusion_matrix
    
    accuracy = accuracy_score(label, pred_probs)
    precision=precision_score(label, pred_probs)
    recall=recall_score(label, pred_probs)
    roc=roc_auc_score(label, pred_probs)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.2f%% " % (precision *100))
    print("Recall: %.2f%% " % (recall * 100))
    print("AUC: %.3f%% " % (roc *100))

    class_report = classification_report(label, pred_probs)
    print(class_report)
    
    plot_confusion_matrix(model_best_params, X_val, y_val, cmap=plt.cm.Blues)  
    plt.show() 