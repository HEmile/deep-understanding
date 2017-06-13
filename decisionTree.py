from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from dataRetriever import retrieveData, get_data
import numpy as np
# import graphviz
# from graphviz import Digraph
import pydotplus
# from IPython.display import Image



def treeClassifier():
    # dtc = DecisionTreeClassifier(max_depth=4, max_features=24)
    dtc = DecisionTreeClassifier(max_depth=4)

    # x, y = retrieveData(make_floats=False)
    x, y = get_data()

    print("len(x)=", len(x))
    print("len(y)=", len(y))

    train_size = 1300

    x_train = x[:train_size]
    y_train = y[:train_size]

    x_test = x[train_size:]
    y_test = y[train_size:]

    dtc.fit(x_train, y_train)
    # dot_data = tree.export_graphviz(dtc, out_file=None) 
    # dot_data = tree.export_graphviz(dtc, out_file="tree.dot", 
    #                      feature_names=["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"],
    #                      class_names=["NO", "YES"], 
    #                      filled=True, rounded=True,  
    #                      special_characters=True)  


    # graph = pydotplus.graph_from_dot_file('tree.dot')
    # graph.write_png("tree.png")

    print("accuracy =", dtc.score(x_test, y_test))

    # print(dtc.decision_path(x_test[0:100]))
    # print(dtc.predict(x_test[0:100]))
    # print(dtc.feature_importances_)

treeClassifier()
# graphviz.view('tree.dot')

