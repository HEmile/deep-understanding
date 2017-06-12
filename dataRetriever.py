# Running retrieveData(file) will return the the rows from 'file'
# as a list, with the exception of the last column, as the first
# returnable. The second is a list of all the rows with only the 
# last column.
import csv
import random

def retrieveData(file_loc="data/UCI_Credit_Card.csv", randomize=True, make_floats=True):
	x = []
	y = []
	with open(file_loc) as data:
		next(data) #skip header
		read_data = csv.reader(data)
		for column in read_data:
			if make_floats:
				x.append(list(map(float, column[1:-1])))
				y.append(float(column[-1]))
			else:
				x.append(column[1:-1])
				y.append(column[-1])
				
	if randomize:
		combined = list(zip(x, y))
		random.shuffle(combined)
		x[:], y[:] = zip(*combined)

	return x, y