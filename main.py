import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import xgboost as xgb
import lightgbm as lgb
import shap
import pickle


def preprocessing(df, dependent, indeps):
    """ Performs basic data preprocessing
        :param df: Vanilla dataset
        :param dependent: Dependent variable
        :param indeps: Independent Variables
        returns: Shuffled and encoded dataset, LabelEncoder object, Dep. feature name
    """
    cols = indeps.copy()
    cols.append(dependent)
    little_df = df[cols]
    little_df[
        ["IP_s1", "IP_s2", "IP_s3", "IP_s4"]
    ] = little_df["Address"].str.split(
        '.', expand=True
    )
    little_df = little_df.drop(columns=["IP_s1", "Address"])
    little_df['Name'] = little_df['Name'].str[:6]
    # Special preprocessing for the Contact column, original data has a pattern
    little_df['Contact'] = little_df[
        'Contact'
    ].str.split('<').str[0].str.strip()
    tryout_df = little_df[[
        "IP_s2", "IP_s3", "Service", "Clus", "Dom",
        "Role", "Name", "Environ", "Contact"
    ]]
    tryout_df['IP_s2'] = tryout_df['IP_s2'].astype(int)
    tryout_df['IP_s3'] = tryout_df['IP_s3'].astype(int)
    df_enc = pd.get_dummies(
        tryout_df, columns=[
            "Service", "Environ", "Dom", "Role", "Name", "Clus"
        ]
    )
    le = LabelEncoder()
    df_enc['Contact'] = le.fit_transform(
        df_enc['Contact']
    )
    df_shf = df_enc.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    with open('lencoder.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)
    return df_shf, le, "Contact"


def rfx_train(df, le, dependent):
    """ Trains and test a Random Forest model 
        :param df: Encoded DataFrame
        :param le: Encoded LabelEncoder object to inverse
        :param dependent: Dependent variable name
        returns clf: The trained model
    """
    y = df[dependent]
    X = df.drop(dependent, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42
    )
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_samples': [0.8, 1.0],
        'criterion': ['gini', 'entropy']
    }

    best_params = {
        'bootstrap': False, 'max_depth': None,
        'max_features': 'log2', 'min_samples_leaf': 1,
        'min_samples_split': 5, 'n_estimators': 260
    }
    # Grid Search operation to find the best params is below, un-docstring when necessary
    """
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    """
    clf = RandomForestClassifier(random_state=42, **best_params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)    
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision)
    print('Recall:', recall)

    #print("Best Parameters:", grid_search.best_params_)
    return clf


def lstm_train(df, le, dep):
    """ Trains and tests the LSTM model
        :param df: Encoded DataFrame
        :param le: Encoded LabelEncoder to inverse
        :param dep: Dependent variable name
    """
    y = df[dep].values
    X = df.drop(dep, axis=1).values
    X = X.astype(float)
    y = le.fit_transform(y)
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.3, verbose=1)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='macro', zero_division=1)
    recall = recall_score(y_test_classes, y_pred_classes, average='macro', zero_division=1)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    

def main():
    """ Driver function """
    df = pd.read_csv("./datasets/prodmail_env.csv")
    dependent = "Contact"
    indeps = [
        "Dom", "Name", "Address", "OSDistro", "Department",
        "Virt/Phy", "Role", "Loc", "Clus",
        "Environ", "Name"         
    ]
    df, le, dep = preprocessing(
        df, dep, indeps
    )
    trained_model = rfx_train(df, le, dep)
    #lstm_train(df, le, dep)
    with open('rfx_model.pkl', 'wb') as file:
        pickle.dump(trained_model, file)


if __name__ == '__main__':
    main()
