

def download_data():
    
    return df

def prepare_data(df):
    
    return df

def feature_engineering(df):
    
    return X, y

def find_best_model(X, y):
    
    return params

def train_model(X, y, params):
    
    return model

def main():
    df=download_data()
    df=prepare_data(df)
    X, y = feature_engineering(df)
    model_params = find_best_model(X, y)
    model = train_model(X, y, model_params)
    