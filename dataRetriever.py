# Running retrieveData(file) will return the the rows from 'file'
# as a list, with the exception of the last column, as the first
# returnable. The second is a list of all the rows with only the 
# last column.
import csv

def retrieveData(file_loc = "data/UCI_Credit_Card.csv"):
	x = []
	y = []
	with open(file_loc) as data:
		next(data) #skip header
		read_data = csv.reader(data)
		for column in read_data:
			x.append(column[:-1])
			y.append(column[-1])
	return x, y