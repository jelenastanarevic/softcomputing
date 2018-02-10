import sys
import numpy as np
import random
import math
import os
import os.path
import nearestNeighbour as NN

from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import SGD

from image import loadImagesVenomous, loadImagesNonvenomous


def readFromFile():

    lines = [line.rstrip('\n') for line in open('snakes.data')]

    trainingData = []
    testData = []
    for line in lines:
        if random.random() < 0.7:
            trainingData.append(line)
        else:
            testData.append(line)

    trainingValues, trainingLabels = prepareDataForNM(stringToDouble(trainingData))
    testValues, testLabels = prepareDataForNM(stringToDouble(testData))

    return[trainingValues, trainingLabels, testValues, testLabels]


def stringToDouble(data):
    retArray = []

    for str in data:
        parts = str.split(',')
        retArray.append([np.double(i) for i in parts])
    return retArray


def prepareDataForNM(data):

    dataValues =[]
    dataLabels = []

    for x in data:
        dataInstance = []
        for y in range(10):
            dataInstance.append(x[y])
        dataValues.append(dataInstance)
        dataLabels.append((x[10],x[11]))

    return [dataValues, dataLabels]


def callNearestNeighbour(trainingValues, trainingLabels, testValues, testLabels):

    preparedTrainingData = NN.prepareData(trainingValues)
    preparedTestData = NN.prepareData(testValues)

    predictions =[]
    k=5
    for x in range(len(preparedTestData)):
        nearestNeighbours = NN.getNeighbours(preparedTrainingData, trainingLabels, preparedTestData[x],k)
        result = NN.getResponse(nearestNeighbours)
        predictions.append(result)
        print('>predicted = ' + repr(result) + ',actual = ' + repr(testLabels[x]))
    accuracy = NN.getAccuracy(preparedTestData, predictions, testLabels)
    print('Accuracy: ' + repr(accuracy) + '%')


def angle_between_points(p0, p1, p2):
    a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
    return math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi


def prepareDataForNeuralNetwork(values):
    retValues = []
    brojac = 0;
    for value in values:
        dist1 = euclideanDistance([value[4], value[5]], [value[2], value[3]], 2) #vrat
        dist2 = euclideanDistance([value[6], value[7]], [value[0], value[1]], 2) #glava
        #angle =  angle_between_points([value[6], value[7]], [value[8], value[9]], [value[0], value[1]])
        retValues.append((dist1/(dist1+dist2), dist2/(dist1+dist2)))
        brojac = brojac + 1
    return retValues


def euclideanDistance(testExample,trainingValue,length):
    distance = 0
    for x in range(length):
        distance+=pow((testExample[x] - trainingValue[x]),2)
    return math.sqrt(distance)


def callNeuralNetwork(trainingValues, trainingLabels, testValues, testLabels):

    preparedTrainingData = prepareDataForNeuralNetwork(trainingValues)
    preparedTestData = prepareDataForNeuralNetwork(testValues)

    trainingv = np.array(preparedTrainingData, np.float32)
    trainingl = np.array(trainingLabels, np.float32)
    testv = np.array(preparedTestData, np.float32)
    testl = np.array(testLabels, np.float32)

    ann = Sequential()
    ann.add(Dense(2, input_shape=(2,), activation='softmax'))
    sgd = SGD(lr=0.0001,momentum=0.9209)
    ann.compile(loss='binary_crossentropy',optimizer=sgd)
    ann.fit(trainingv, trainingl, epochs=1, batch_size=len(trainingv))

    results = ann.predict(testv)
    results = np.array(results).tolist()
    roundResults = [(round(x[0]), round(x[1])) for x in results]
    for xx in range(len(preparedTestData)):
        print('>predicted = ' + (repr(roundResults[xx])) + ', ' + (repr(results[xx])) + ',actual = ' + repr(testLabels[xx]))
    scores = ann.evaluate(testv, testl, verbose=0)
    print('Accuracy: ' + repr(scores) + '%')

    ann.save("ann.h5py")


def loadNeuralNetwork(testValues, testLabels):
    ann = load_model("ann.h5py")
    preparedTestData = prepareDataForNeuralNetwork(testValues)
    testv = np.array(preparedTestData, np.float32)
    testl = np.array(testLabels, np.float32)

    results = ann.predict(testv)
    results = np.array(results).tolist()
    roundResults = [(round(x[0]), round(x[1])) for x in results]
    for xx in range(len(preparedTestData)):
        print('>predicted = ' + (repr(roundResults[xx])) + ', ' + (repr(results[xx])) + ',actual = ' + repr(testLabels[xx]))
    scores = ann.evaluate(testv, testl, verbose=0)
    print('Accuracy: ' + repr(scores) + '%')


def main():

    while(1):
        print('Select one of the following options: ')
        print('>> 1. Load training and validation data ')
        print('>> 2. Use the nearest neighbour algorithm to predict results ')
        print('>> 3. Fit neural network and use it to predict results')
        print('>> 4. Use existing neural network it to predict results')
        print('>> 5. Exit the program')

        try:
            line = int(sys.stdin.readline())
            if (line == 1):
                file = open('snakes.data', 'w')
                loadImagesVenomous(file)
                loadImagesNonvenomous(file)
                file.close()
                print('Done.')
            elif(line == 2):
                if (not(os.path.isfile('snakes.data')) or (os.path.getsize('snakes.data') == 0)):
                    print('Load data!')
                else:
                    trainingValues, trainingLabels, testValues, testLabels = readFromFile()
                    callNearestNeighbour(trainingValues, trainingLabels, testValues, testLabels)
            elif(line==3):
                if (not (os.path.isfile('snakes.data')) or (os.path.getsize('snakes.data') == 0)):
                    print('Load data!')
                else:
                    trainingValues, trainingLabels, testValues, testLabels = readFromFile()
                    callNeuralNetwork(trainingValues, trainingLabels, testValues, testLabels)
            elif(line==4):
                if (not (os.path.isfile('snakes.data')) or (os.path.getsize('snakes.data') == 0)):
                    print('Load data!')
                else:
                    trainingValues, trainingLabels, testValues, testLabels = readFromFile()
                    loadNeuralNetwork(testValues, testLabels)
            elif(line==5):
                break
            else:
                print('Incorrect input option')
        except ValueError as verr:
            print('Error.')
        except Exception as xec:
            print('Error.')


main()