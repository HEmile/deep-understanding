from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from dataRetriever import retrieveData, get_data, get_reged_data
import numpy as np
# import graphviz
import pydotplus
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

###dengue data
def treeRegressor():
    dtr = DecisionTreeRegressor(max_depth=4)
    # dtr = DecisionTreeRegressor()
    x, y = get_data()

    train_size = 1300

    x_train = x[:train_size]
    y_train = y[:train_size]

    x_test = x[train_size:]
    y_test = y[train_size:]

    dtr.fit(x_train, y_train)
    dot_data = tree.export_graphviz(dtr, out_file="tree.dot", 
                        feature_names=["ndvi_ne","ndvi_nw","ndvi_se","ndvi_sw","precipitation_amt_mm","reanalysis_air_temp_k","reanalysis_avg_temp_k","reanalysis_dew_point_temp_k","reanalysis_max_air_temp_k","reanalysis_min_air_temp_k","reanalysis_precip_amt_kg_per_m2","reanalysis_relative_humidity_percent","reanalysis_sat_precip_amt_mm","reanalysis_specific_humidity_g_per_kg","reanalysis_tdtr_k","station_avg_temp_c","station_diur_temp_rng_c","station_max_temp_c","station_min_temp_c","station_precip_mm"], 
                        filled=True, rounded=True,  
                        special_characters=True)

    # graph = pydotplus.graph_from_dot_file('tree.dot')
    # graph.write_png("tree.png")

    # print("accuracy =", dtr.score(x_test, y_test))

    predictions = dtr.predict(x_test)

    n = len(y_test)
    good_prediction = 0
    sum_of_squares = 0
    for prediction, y in zip(predictions, y_test):
        sum_of_squares += np.power((prediction-y), 2)
        # print(prediction, "---", y)
        if np.abs(prediction - y) < 10:
            good_prediction+= 1

    print("\"accuracy\"=", good_prediction/n)
    print("mean sum of squares=", sum_of_squares/n)



###CreditCard data
def treeClassifier():
    dtc = DecisionTreeClassifier(max_depth=4)

    x, y = retrieveData(make_floats=False)

    train_size = 25000

    x_train = x[:train_size]
    y_train = y[:train_size]

    x_test = x[train_size:]
    y_test = y[train_size:]

    dtc.fit(x_train, y_train)
    dot_data = tree.export_graphviz(dtc, out_file="tree.dot", 
                         feature_names=["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"],
                         class_names=["NO", "YES"], 
                         filled=True, rounded=True,  
                         special_characters=True)  


    graph = pydotplus.graph_from_dot_file('tree.dot')
    graph.write_png("tree.png")

    print("accuracy =", dtc.score(x_test, y_test))

    # print(dtc.decision_path(x_test[0:100]))
    # print(dtc.predict(x_test[0:100]))
    # print(dtc.feature_importances_)

def regedTreeClassifier():
    x, y = get_reged_data()

    x_array = np.array(x)

    hot_x = [[]] * x_array.shape[0]

    for column in range(x_array.shape[1]):
        label_enc = LabelEncoder()
        label_encoder = label_enc.fit(x_array[:, column])
        integer_classes = label_encoder.transform(label_encoder.classes_).reshape(len(label_encoder.classes_), 1)

        hot_enc = OneHotEncoder()
        one_hot_encoder = hot_enc.fit(integer_classes)

        num_of_rows = x_array.shape[0]
        t = label_encoder.transform(x_array[:, column]).reshape(num_of_rows, 1)

        new_features = one_hot_encoder.transform(t)

        # x_array = np.concatenate([x_array, new_features.toarray()], axis = 1)
        hot_x = np.append(hot_x, new_features.toarray(), axis = 1)

        print("onehot-ting column", column)

    # hot_x = []
    # i = 0
    # for row in x:
    #     hot_row = []
    #     for value in row:
    #         oneHotValue = [0] * 1000
    #         oneHotValue[int(value)] = 1
    #         hot_row += oneHotValue
    #     i+=1
    #     print(i)


    #     hot_x.append(hot_row)

    dtc = DecisionTreeClassifier(max_depth=4)

    train_size = 400

    x_train = hot_x[:train_size]
    y_train = y[:train_size]

    x_test = hot_x[train_size:]
    y_test = y[train_size:]

    dtc.fit(x_train, y_train)

    print("accuracy =", dtc.score(x_test, y_test))

    dot_data = tree.export_graphviz(dtc, out_file="tree_reged.dot", 
                    filled=True, rounded=True,  
                    special_characters=True)


    graph = pydotplus.graph_from_dot_file('tree_reged.dot')
    graph.write_png("tree_reged.png")
    print("saved tree to:", "tree_reged.png")


    # print(dtc.decision_path(x_test[0:100]))
    print(dtc.predict(x_test))
    # print(dtc.feature_importances_)


# treeClassifier()
# treeRegressor()
regedTreeClassifier()