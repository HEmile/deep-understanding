from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from dataRetriever import retrieveData, get_data
import numpy as np
# import graphviz
import pydotplus

###dengue data
def treeRegressor():
    dtr = DecisionTreeRegressor(max_depth=4)
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

    graph = pydotplus.graph_from_dot_file('tree.dot')
    graph.write_png("tree.png")

    print("accuracy =", dtr.score(x_test, y_test))


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

# treeClassifier()
treeRegressor()