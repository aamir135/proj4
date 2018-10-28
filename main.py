import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import time

def main ():
    DataSetSelection = input("Please Enter the name of DataSet (housing or power): \n")
    print("------Note: noraml-eq method only works with housing dataset--------")
    Selection = input("Please Enter the name from  the list (lr or ransac or ridge or lasso or normal-eq or non-linear: \n")
    start_time = time.time()
    if (DataSetSelection == "housing"):
        dataset = pd.read_csv('D:\ML_Project4\housing.csv', header=0)
        X = dataset.iloc[:, 0:12].values
        y = dataset.iloc[:, 13].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        if (Selection == "lr"):
            model = LinearRegression(normalize=False)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            #mean square error
            error_train = mean_squared_error(y_train, y_train_pred)
            error_test = mean_squared_error(y_test, y_test_pred)
            print('MSE train: %3f, test: %3f' % (error_train, error_test))
            # R^2
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            print('R^2 train: %3f, test: %3f' % (r2_train, r2_test))

        elif (Selection == "ransac"):
            model = RANSACRegressor(random_state=1)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            error_train = mean_squared_error(y_train, y_train_pred)
            error_test = mean_squared_error(y_test, y_test_pred)
            print('MSE train: %3f, test: %3f' % (error_train, error_test))
            # R^2
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            print('R^2 train: %3f, test: %3f' % (r2_train, r2_test))

        elif (Selection == "ridge"):

            model = Ridge(alpha= 1 , normalize=True)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            error_train = mean_squared_error(y_train, y_train_pred)
            error_test = mean_squared_error(y_test, y_test_pred)
            print('MSE train: %3f, test: %3f' % (error_train, error_test))
            # R^2
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            print('R^2 train: %3f, test: %3f' % (r2_train, r2_test))


        elif (Selection == "lasso"):
            model = Lasso(alpha= 1 , normalize=True)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            error_train = mean_squared_error(y_train, y_train_pred)
            error_test = mean_squared_error(y_test, y_test_pred)
            print('MSE train: %3f, test: %3f' % (error_train, error_test))
            # R^2
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            print('R^2 train: %3f, test: %3f' % (r2_train, r2_test))

        elif (Selection == "normal-eq"):
            #X = dataset.iloc[:, 0:12].values
            #y = dataset.iloc[:, 13].values
            # Normal Equation
            onevec = np.ones((X.shape[0]))
            #print(onevec)
            onevec = onevec[:, np.newaxis]
            #print(onevec)
            Xb = np.hstack((onevec, X))
            #print(Xb)
            w = np.zeros(X.shape[1])
            z = np.linalg.inv(np.dot(Xb.T, Xb))
            w = np.dot(z, np.dot(Xb.T, y))
            print('Slope: %3f' % w[1])
            print('Intercept: %3f' % w[0])


        elif (Selection == "non-linear"):
            poly = PolynomialFeatures(degree=2)
            X2 = poly.fit_transform(X)
            #print(X2)
            poly_r = LinearRegression()
            poly_r.fit(X2, y)
            y_poly_pred = poly_r.predict(X2)
            error_poly = mean_squared_error(y, y_poly_pred)
            r2_poly = r2_score(y, y_poly_pred)
            print('MSE : %3f' % (error_poly))
            print('R^2 : %3f' % (r2_poly))
        else:
            print("input right selection")
    elif (DataSetSelection == "power"):
        dataset = pd.read_csv('D:\ML_Project4\solar_deleted_columns.csv', header=0)
        X = dataset.iloc[:, 0:4].values
        y = dataset.iloc[:, 5].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        if (Selection == "lr"):
            model = LinearRegression(normalize= False)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # mean square error
            error_train = mean_squared_error(y_train, y_train_pred)
            error_test = mean_squared_error(y_test, y_test_pred)
            print('MSE train: %3f, test: %3f' % (error_train, error_test))
            # R^2
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            print('R^2 train: %3f, test: %3f' % (r2_train, r2_test))

        elif (Selection == "ransac"):
            model = RANSACRegressor(random_state=1)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            error_train = mean_squared_error(y_train, y_train_pred)
            error_test = mean_squared_error(y_test, y_test_pred)
            print('MSE train: %3f, test: %3f' % (error_train, error_test))
            # R^2
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            print('R^2 train: %3f, test: %3f' % (r2_train, r2_test))

        elif (Selection == "ridge"):
            model = Ridge(alpha=1, normalize=True)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            error_train = mean_squared_error(y_train, y_train_pred)
            error_test = mean_squared_error(y_test, y_test_pred)
            print('MSE train: %3f, test: %3f' % (error_train, error_test))
            # R^2
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            print('R^2 train: %3f, test: %3f' % (r2_train, r2_test))

        elif (Selection == "lasso"):
            model = Lasso(alpha=1, normalize=True)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            error_train = mean_squared_error(y_train, y_train_pred)
            error_test = mean_squared_error(y_test, y_test_pred)
            print('MSE train: %3f, test: %3f' % (error_train, error_test))
            # R^2
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            print('R^2 train: %3f, test: %3f' % (r2_train, r2_test))

        elif (Selection == "non-linear"):
            poly = PolynomialFeatures(degree=2)
            X2 = poly.fit_transform(X)
            # print(X2)
            poly_r = LinearRegression()
            poly_r.fit(X2, y)
            y_poly_pred = poly_r.predict(X2)
            error_poly = mean_squared_error(y, y_poly_pred)
            r2_poly = r2_score(y, y_poly_pred)
            print('MSE : %3f' % (error_poly))
            print('R^2 : %3f' % (r2_poly))
        else:
            print("input right selection")
    else:
        print("please enter right parameters")
    print("Running time = %s seconds " % (time.time() - start_time))
main()