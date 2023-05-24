def problem3(directory):
    """
    """
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model.logistic import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # change scaler ?
    from sklearn.compose import ColumnTransformer
    from sklearn.neighbors import KNeighborsClassifier
    
    raw_data=pd.read_csv(directory,na_values=[" ", ""]).dropna()
    data=raw_data.drop("customerID", axis=1)
    cat_columns= data.drop(["tenure","MonthlyCharges", "TotalCharges", "Churn"], axis=1).columns
    num_columns=["tenure","MonthlyCharges", "TotalCharges"]
    transformer=ColumnTransformer([("dummy",OneHotEncoder(),cat_columns),("standart_scaler",StandardScaler(),num_columns)])
    a_data=transformer.fit_transform(data)
    raw_data["Churn"]=raw_data.Churn.map(lambda x: 1 if "Yes" in x else 0)
    
    scaler=MinMaxScaler()# change scaler ?
    data_scaled=scaler.fit_transform(a_data)
    X=data_scaled
    y=raw_data["Churn"]
    X_train,X_test,y_train,y_test=train_test_split(data_scaled,y,shuffle=True)
    
    kf=KFold(n_splits=5,shuffle=True)
    models=[LogisticRegression(), DecisionTreeClassifier(), LinearSVC(), KNeighborsClassifier(), MLPClassifier()]
    models_string=["LogisticRegression", "DecisionTree", "LinearSVC", "KNN", "MLPClassifier"]

    def scores_calc(model, X, y, trains, kfold):
        """
        Calculates average accuracy scores for training and test splits of cross validation
        Parameters:
            model : function that implements fit and score
            X : scaled data
            y : values
            trains : train data in form (X_train, y_train)
            kfold : KFold value
        Returns : tuple of form (<training scores>, <test scores>)
        """
        model.fit(trains[0],trains[1])
        test_scores = cross_val_score(model,X,y,cv=kf,scoring='accuracy').mean()
        training_scores = model.score(trains[0],trains[1])
        return((training_scores,test_scores))
    
    rows = []
    i = 0
    for model in models:
        scores = scores_calc(model, X, y, (X_train, y_train), kf)
        rows[i] = [models_string[i], scores[0], scores[1]]
        i += 1
    df = pd.DataFrame(np.array(rows), columns=["model", "train", "test"])
    display(df)
