import datetime
from pymongo import MongoClient
import scrapy
import numpy
import subprocess
import re
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dense, Masking, LSTM

def trainModel_1(trainDataFilename):
	numpy.random.seed(11)

	datasetX = numpy.loadtxt(trainDataFilename, delimiter = ",", dtype = "float", usecols = range(2, 6))
	datasetY = numpy.loadtxt(trainDataFilename, delimiter = ",", dtype = "float", usecols = [5])
	datasetX = datasetX[(datasetX != -1).all(axis = 1), :]
	datasetX = datasetX[:-1]
	datasetY = datasetY[datasetY > -1]
	datasetY = datasetY[1:]

	model = Sequential()
	model.add(Dense(6, input_dim = 4, activation = "relu"))
	model.add(Dense(4, activation = "relu"))
	model.add(Dense(1, activation = "sigmoid"))

	model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])

	model.fit(datasetX, datasetY, validation_split = 0.2, epochs = 150, batch_size = 5)

	scores = model.evaluate(datasetX, datasetY)
	print("")
	print("{}: {}".format(model.metrics_names[1], scores[1]*100))

	return model

def trainModel_2(trainDataFilename):
	numpy.random.seed(11)

	datasetX = numpy.loadtxt(trainDataFilename, delimiter = ",", dtype = "float", usecols = range(2, 6))
	datasetY = numpy.loadtxt(trainDataFilename, delimiter = ",", dtype = "float", usecols = [5])
	datasetX = datasetX[(datasetX != -1).all(axis = 1), :]
	datasetX = datasetX[:-1]
	datasetY = datasetY[datasetY > -1]
	datasetY = datasetY[1:]

	model = Sequential()
	model.add(Dense(8, input_dim = 4, activation = "relu"))
	model.add(Dense(3, activation = "relu"))
	model.add(Dense(1, activation = "sigmoid"))

	model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])

	model.fit(datasetX, datasetY, validation_split = 0.2, epochs = 150, batch_size = 10)

	scores = model.evaluate(datasetX, datasetY)
	print("")
	print("{}: {}".format(model.metrics_names[1], scores[1]*100))

	return model

def trainModel_3(trainDataFilename):
	numpy.random.seed(117)

	data = numpy.loadtxt(trainDataFilename, delimiter = ",", dtype = "float", usecols = range(2, 6))
	datasetX, datasetY = [], []
	for i in range(len(data) - 3):
		datasetX.append([])
		datasetY.append(data[i+3][3])
		for j in range(0, 1):
			datasetX[i].append(data[i])
			datasetX[i].append(data[i+1])
			datasetX[i].append(data[i+2])

	datasetX = numpy.array(datasetX)
	datasetY = numpy.array(datasetY)

	model = Sequential()
	model.add(Masking(mask_value = -1, input_shape = (3,4)))
	model.add(LSTM(1))
	model.add(Dense(1, activation = "relu"))

	model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])
	model.fit(datasetX, datasetY, validation_split = 0.2, epochs = 700, batch_size = 1)

	scores = model.evaluate(datasetX, datasetY)
	print("")
	print("{}: {}".format(model.metrics_names[1], scores[1]*100))

	return model

def trainModel_4(trainDataFilename):
	numpy.random.seed(117)

	data = numpy.loadtxt(trainDataFilename, delimiter = ",", dtype = "float", usecols = range(2, 6))
	data = data[(data != -1).all(axis = 1), :]
	datasetX, datasetY = [], []
	for i in range(len(data) - 3):
		datasetX.append([])
		datasetY.append(data[i+3][3])
		for j in range(0, 1):
			datasetX[i].append(data[i])
			datasetX[i].append(data[i+1])
			datasetX[i].append(data[i+2])

	datasetX = numpy.array(datasetX)
	datasetY = numpy.array(datasetY)

	model = Sequential()
	model.add(LSTM(3, return_sequences = True, input_shape = (3, 4)))
	model.add(LSTM(4, return_sequences = True))
	model.add(LSTM(5, return_sequences = True))
	model.add(LSTM(3))
	model.add(Dense(1, activation = "relu"))

	model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])

	model.fit(datasetX, datasetY, validation_split = 0.2, epochs = 150, batch_size = 10)

	scores = model.evaluate(datasetX, datasetY)
	print("")
	print("{}: {}".format(model.metrics_names[1], scores[1]*100))

	return model

def trainModel_5(trainDataFilename):
	numpy.random.seed(11)

	datasetX = numpy.loadtxt(trainDataFilename, delimiter = ",", dtype = "float", usecols = range(2, 6))
	datasetY = numpy.loadtxt(trainDataFilename, delimiter = ",", dtype = "float", usecols = [5])
	datasetX = datasetX[(datasetX != -1).all(axis = 1), :]
	datasetX = datasetX[:-1]
	datasetY = datasetY[datasetY > -1]
	datasetY = datasetY[1:]

	model = Sequential()
	model.add(Dense(12, input_dim = 4, activation = "relu"))
	model.add(Dense(8, activation = "relu"))
	model.add(Dense(4, activation = "relu"))
	model.add(Dense(1, activation = "sigmoid"))

	model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])

	model.fit(datasetX, datasetY, validation_split = 0.2, epochs = 150, batch_size = 5)

	scores = model.evaluate(datasetX, datasetY)
	print("")
	print("{}: {}".format(model.metrics_names[1], scores[1]*100))

	return model

def saveModel(model, modelFilenamePrefix):
	structFilename = modelFilenamePrefix + ".json"
	model_json = model.to_json()
	with open(structFilename, "w") as f:
		f.write(model_json)

	weightFilename = modelFilenamePrefix + ".h5"
	model.save_weights(weightFilename)

def readModel(modelFilenamePrefix):
	structFilename = modelFilenamePrefix + ".json"
	with open(structFilename, "r") as f:
		model_json = f.read()
	model = model_from_json(model_json)

	weightFilename = modelFilenamePrefix + ".h5"
	model.load_weights(weightFilename)

	return model

def predictNewDataset(newX, model):
	newY = model.predict(newX, batch_size=10)
	#numpy.savetxt(newTargetFilename, newY, delimiter=",", fmt="%.10f")
	return newY

def collection():
	#subprocess.run("scrapy startproject CourseDM", shell = True)
	db.courses.drop()
	print("Collection dropping and empty collection creating are successful.")

def crawl():
	url = input('Please enter a URL for data crawling: ')
	if url == 'default':
		url = 'http://comp4332.com/realistic'
	with open('url.txt', 'w') as f:
		f.write(url)
	subprocess.run('scrapy crawl mongo', shell=True)

def search(): 
	while True:
		try:
			s_method = int(input("1. By Keyword\n2. By Waitlist Size\n3. Back to main menu."))
		except ValueError:
			print("Error. Please enter between 1 to 3.")
		else:
			if s_method != 1 and s_method != 2 and s_method != 3:
				print("Error. Please enter between 1 to 3.")
				continue
			else:
				if s_method == 1:
					key = input("Enter the keyword(s).")
					k1 = ".*"+key.split( )[0]+".*"
					k2 = ".*"+key.split( )[1]+".*"
					### searching
					db = client["university"]
					db.courses.aggregate([
						{"$unwind": "$sections"},
						{"$sort": {"code": 1, "sections.sectionId": 1, "sections.recordTime": -1}},
						{
							"$group": {
								"_id": {"code": "$code", "sectionId": "$sections.sectionId"}, 
								"code": {"$first": "$code"}, "semester":{"$first": "$semester"}, "title": {"$first": "$title"}, "credits": {"$first": "$credits"},
								"description": {"$first": "$description"}, "recordTime": {"$first": "$sections.recordTime"}, "sectionId": {"$first": "$sections.sectionId"},
								"offerings": {"$push": "$sections.offerings"}, "quota": {"$first": "$sections.quota"}, "enrol": {"$first": "$sections.enrol"}, "avail": {"$first": "$avail"}, 
								"wait": {"$first": "$sections.wait"}, "remarks": {"$push": "$sections.remarks"}
							}
						},
						{"$out": "tempOutput"}
					], allowDiskUse = True)
					
					R1 = list(db.tempOutput.find({"$or": [{"title": {"$regex": k1}}, {"title": {"$regex": k2}}]},
						{
							"code": 1, "title": 1, "credits": 1, 
							"recordTime": 1, "sectionId": 1, 
							"quota": 1, "enrol": 1, "avail": 1, "wait": 1,
							"_id": 0
						}
					))

					R2 = list(db.tempOutput.find({"$or": [{"description": {"$regex": k1}}, {"description": {"$regex": k2}}]},
						{
							"code": 1, "title": 1, "credits": 1, 
							"recordTime": 1, "sectionId": 1, 
							"quota": 1, "enrol": 1, "avail": 1, "wait": 1,
							"_id": 0
						}
					))

					#R3 = list(db.tempOutput.find({"$or": [{"remarks": {"$regex": k1}}, {"remarks": {"$regex": k2}}]},
						#{
							#"code": 1, "title": 1, "credits": 1, 
							#"recordTime": 1, "sectionId": 1,
							#"quota": 1, "enrol": 1, "avail": 1, "wait": 1, "remarks": 1,
							#"_id": 0
						#}
					#))

					db.temp.insert({"col1": R1, "col2": R2})

					cursor = db.temp.aggregate([
						{"$project": {"answer": {"$setUnion": ["$col1", "$col2"]}, "_id": 0}}
					], allowDiskUse = True)

					for oneCourse in cursor:
						print(oneCourse)

					db.temp.drop()
					db.tempOutput.drop()

					print("Search sucessful")
					continue

				elif s_method == 2:
					start_ts = input("Starting Time Slot? ")
					start_var = datetime.datetime.strptime(start_ts, "%Y-%m-%d %H:%M")

					end_ts = input("Ending Time Slot? ")
					end_var = datetime.datetime.strptime(end_ts, "%Y-%m-%d %H:%M")
					while True:
						try:
							f = float(input("Minimum ratio of waitlist students to students enrolled to that time slot "))
						except ValueError:
							print("Error. Please enter a non-negative number.")
						else:
							if f < 0:
								print("Error. Please enter a non-negative number.")
								continue
							else:
								break

					### searching
					db = client["university"]
					db.courses.aggregate([
						{"$unwind": "$sections"},
						{
							"$project": {
								"code": 1, "semester": 1, "title": 1, "credits": 1, "description": 1,
								"recordTime": "$sections.recordTime", "sectionId": "$sections.sectionId", "offerings": "$sections.offerings",
								"quota": "$sections.quota", "enrol": "$sections.enrol", "avail": "$sections.avail",
								"wait": "$sections.wait", "remarks": "$sections.remarks",
								"satisfied": {"$gte": ["$sections.wait", {"$multiply": ["$sections.enrol", f]}]}
							}
						},
						{
							"$match": {
								"$and":[
									{"recordTime": {"$gte": start_var}},#("2018-02-01T11:00:00Z", "%Y-%m-%dT%H:%M:%SZ")}},
									{"recordTime": {"$lte": end_var}}#("2018-02-02T11:00:00Z", "%Y-%m-%dT%H:%M:%SZ")}}
								]
							}
						},
						{ 
							"$project": {
								"code": 1, "title": 1, "credits": 1, "description": 1,
								"recordTime": 1, "sectionId": 1, "offerings": 1,
								"quota": 1, "enrol": 1, "avail": 1, "wait": 1, "remarks": 1,
								"satisfied": 1
							}
						},
						{"$sort": {"code": 1, "sectionId": -1, "recordTime": -1}}, # unexpectedly, section: -1 in order to sort ascendingly
						{
							"$group": {
								"_id": "$_id", 
								"code": {"$first": "$code"}, "title": {"$first": "$title"}, "credits": {"$first": "$credits"}, 
								"description": {"$first": "$description"}, "satisfyArray": {"$push": "$satisfied"},
								"sections": {"$push": "$sections"}
								#"_time_slot": {"$first": "$SecList_Taken.ts"},
								#"_section": {"$first": "$SecList_Taken.section"},"_DateTime": {"$first": "$SecList_Taken.DT"},
								#"_quota": {"$first": "$SecList_Taken.quota"}, "_enrol": {"$first": "$SecList_Taken.enrol"}, 
								#"_avail": {"$first": "$SecList_Taken.avail"}, "_wait": {"$first": "$SecList_Taken.wait"},
								#"_Satisfied": {"$first": "$Satisfied"}
							}
						},
						{"$project": {"_id": 0}},
						{"$out": "tempOutput"}
					], allowDiskUse = True)

					cursor2 = db.tempOutput.find({"satisfyArray": {"$in": [True]}}, {"_id": 0, "satisfyArray": 0})
					for oneCourse in cursor2:
						print(oneCourse)

					db.temp.drop()
					db.tempOutput.drop()

					print("Search sucessful")
					continue
				elif s_method == 3:
					break

def size_pred():
	cc = input("Course Code? ")
	ts = input("Starting Time Slot? ")
	ln = int(input("Lecture Number? "))
	newFilename = cc+"-L"+str(ln)+".csv"
	newData_str = numpy.loadtxt(newFilename, delimiter = ",", dtype = "str", usecols = range(0, 2))
	newData_str = numpy.array(newData_str)
	#rowNum = int(newData_str[(newData_str == ts).any(axis = 1)][0][1])
	rowNum = int(numpy.where(newData_str[:, 0] == ts)[0])
	newData = numpy.loadtxt(newFilename, delimiter = ",", dtype = "float", usecols = range(2, 6))
	newX_lookback1 = newData[rowNum - 1: rowNum]
	newX_lookback3 = newData[rowNum - 3: rowNum]
	new_datasetX, new_datasetY = [], []
	for i in range(len(newData) - 3):
		new_datasetX.append([])
		new_datasetY.append(newData[i+3][3])
		for j in range(0, 1):
			new_datasetX[i].append(newData[i])
			new_datasetX[i].append(newData[i+1])
			new_datasetX[i].append(newData[i+2])

	new_datasetX = numpy.array(new_datasetX)
	#print(new_datasetX)

	outArr = [0]*5
	for i in range(0, 2):	
		readModelName = cc+"-L"+str(ln)+"-model_"+str(i+1)
		model = readModel(readModelName)
		result = predictNewDataset(newX_lookback1, model)
		outArr[i] = int(result)
	for i in range(2, 4):
		readModelName = cc+"-L"+str(ln)+"-model_"+str(i+1)
		model = readModel(readModelName)
		rArr = predictNewDataset(new_datasetX, model)
		outArr[i] = int(rArr[rowNum, 0])

	readModelName = cc+"-L"+str(ln)+"-model_"+str(5)
	model = readModel(readModelName)
	result = predictNewDataset(newX_lookback1, model)
	outArr[4] = int(result)

	print(outArr)
	print("Predict sucessful")
	return outArr

def size_train():

	### Training
	db.courses.aggregate([
	{"$match": {
		"$or": [
			{"code": "COMP1942"}, {"code": "COMP4211"}, {"code": "COMP4221"}, {"code": "COMP4321"}, {"code": "COMP4331"}, {"code": "COMP4332"},
			{"code": "RMBI1010"}, {"code": "RMBI3000A"}, {"code": "RMBI4210"}, {"code": "RMBI4310"}
		]}
	},
	{"$out": "tempOutput"}
	], allowDiskUse = True)
	
	restrictedCourse = ['COMP1942-L1', 'COMP4211-L1', 'COMP4221-L1', 'COMP4321-L1', 'COMP4331-L1', 'COMP4332-L1', 'RMBI1010-L1', 'RMBI3000A-L1', 'RMBI4210-L1', 'RMBI4310-L1']
	for rCourse in restrictedCourse:
		cid = rCourse.split("-")[0]
		secId = rCourse.split("-")[1]
		stats = [[0 for x in range(4)] for y in range(992)] 

		for i in range(0, 992):
			timeVar = datetime.datetime.strptime(Timestamp[i][0], "%Y-%m-%d %H:%M")
			listOfsecCourse = db.tempOutput.aggregate([
				{"$match": {"code": cid}},
				{"$unwind": "$sections"},
				{"$match": {
					"$and": [
						{"sections.sectionId": secId},
						{"sections.recordTime": timeVar}
					]}
				},
				{"$project": {"_id": 0, "quota": "$sections.quota", "enrol": "$sections.enrol", "avail": "$sections.avail", "wait": "$sections.wait"}}
			], allowDiskUse = True)
		
			for secCourse in listOfsecCourse:
				if secCourse != None:
					stats[i][0] = secCourse["quota"]
					stats[i][1] = secCourse["enrol"]
					stats[i][2] = secCourse["avail"]
					stats[i][3] = secCourse["wait"]
				else:
					for j in range(0, 4):
						stats[i][j] = -1

		newA = numpy.column_stack((Timestamp, numpy.array(stats)))
		for i in range(0, 992):
			if newA[i][2] == "0":
				newA[i][2] = "-1"
				newA[i][3] = "-1"
				newA[i][4] = "-1"
				newA[i][5] = "-1"

		filename = rCourse+".csv"
		numpy.savetxt(filename, newA, delimiter = ",", fmt = "%s")
		model1 = trainModel_1(filename)
		saveModel(model1, rCourse+"-model_"+str(1))
		model2 = trainModel_2(filename)
		saveModel(model2, rCourse+"-model_"+str(2))
		#model3 = trainModel_3(filename)
		#saveModel(model3, rCourse+"-model_"+str(3))
		model4 = trainModel_4(filename)
		saveModel(model4, rCourse+"-model_"+str(4))
		#model5 = trainModel_5(filename)
		#saveModel(model5, rCourse+"-model_"+str(5))

	db.tempOutput.drop()

	print("Waiting list size training is successful.")


client = MongoClient("mongodb://localhost:27017")
db = client["university"]
print("Hello. Welcome to use the service.\nPlease choose an option.")

Timestamp = []
dateTimeVariable = datetime.datetime.strptime("2018-01-25 08:30", "%Y-%m-%d %H:%M")
for i in range(0, 992):
	dateTimeVariable = dateTimeVariable + datetime.timedelta(minutes = 30)
	plus_30_str = dateTimeVariable.strftime("%Y-%m-%d %H:%M")
	Timestamp.append([])
	for j in range(0, 1):
		Timestamp[i].append(plus_30_str)
		Timestamp[i].append(i)

output = []
while True:
	try:
		print("\n1. Collection Dropping and Empty Collection Creating\n2. Data Crawling\n3. Course Search\n4. Waiting List Size Prediction\n5. Waiting List Size Training\n6. Quit the program ")
		num = int(input("Please enter between 1 to 6: "))
	except ValueError:
		print("Error. Please enter between 1 to 6. ")
	else:
		if num < 1 or num > 6:
			print("Error. Please enter between 1 to 6. ")
			continue
		else: 
			if num == 1:
				collection()
				continue
			elif num == 2:
				crawl()
				continue
			elif num == 3:
				search()
				continue
			elif num == 4:
				output = size_pred()
				output
				continue
			elif num == 5:
				size_train()
				continue
			elif num == 6:
				# client.close()
				break

#if __name__ == '__main__':
	#main()