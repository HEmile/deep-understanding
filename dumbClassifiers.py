import random
from dataRetriever import retrieveData

def randomClassifier(data_loc = "data/UCI_Credit_Card.csv", n = 5000):
	_, y = retrieveData(data_loc)
	y_test = y[-n:]

	true_positive = 0
	for y in y_test:
	    classification = random.randint(0,1)
	    # classification = 0
	    if int(y) == classification:
	        true_positive += 1

	print("k = random, accuracy =", true_positive/n)

def kClassifier(k = 0, data_loc = "data/UCI_Credit_Card.csv", n = 5000):
	_, y = retrieveData(data_loc)
	y_test = y[-n:]

	true_positive = 0
	for y in y_test:
	    if int(y) == k:
	        true_positive += 1

	print("k =", str(k) + ", accuracy =", true_positive/n)
