import math
import operator

def prepareData(values):
    retValues = []
    for value in values:
        dist1 = euclideanDistance([value[4],value[5]],[value[2],value[3]],2)
        dist2 = euclideanDistance([value[6], value[7]], [value[0], value[1]], 2)
        retValues.append((dist1 / (dist1 + dist2), dist2 / (dist1 + dist2)))
    return retValues


def euclideanDistance(testExample,trainingValue,length):
    distance = 0
    for x in range(length):
        distance+=pow((testExample[x] - trainingValue[x]),2)
    return math.sqrt(distance)


def getNeighbours(trainingValues, trainingLabels, testExample,k):
    distances=[]
    length = len(testExample)

    for x in range(len(trainingValues)):
        dist = euclideanDistance(testExample,trainingValues[x],length)
        distances.append((trainingValues[x],trainingLabels[x],dist))
    distances.sort(key=operator.itemgetter(2))

    neighbours = []
    for kk in range(k):
        neighbours.append((distances[kk][0],distances[kk][1]))
    return neighbours


def getResponse(nearestNeighbours):
    classVotes={}

    for x in range(len(nearestNeighbours)):
        response = nearestNeighbours[x][1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions, testLabels):
    correct = 0
    for x in range(len(testSet)):
        if testLabels[x] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0