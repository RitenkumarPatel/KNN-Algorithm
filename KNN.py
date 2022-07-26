from math import sqrt
from random import seed
from random import randrange
from csv import reader

#Load a CSV(Comma seperated value) file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:         #Removes any blank values
                continue
            dataset.append(row)
        return dataset

#Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())    #.strip() removes spaces before and after the message

#Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]     #List of each value from each row (in this case the species of the flower as a string)
    unique = set(class_values)              #Saves only the unique species (sets can have no repeat values)
    lookup = dict()                 #creating a lookup dictionary
    for i, value in enumerate(unique):          #for each value in unique, create a lookup value that associates a string with an integer
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]          #set each value (of the species) equal to the number(via the lookup)
    return lookup

#Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

#Rescale dataset columns to the range 0 to 1
def nomralize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0])/ (minmax[i][1] - minmax[i][0])     #(value in the cell - minvalue in that column)/(maxValue in Column - minValue in Column)

#Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))        #sets index to a random number within the length of the ((copy of) original dataset
            fold.append(dataset_copy.pop(index))        #takes that random index from the copy and puts it into the fold
        dataset_split.append(fold)
    return dataset_split

#Calculate accuracy percentage based on how many correct predictions were made
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

#Evaluate an algorithm using a cross validation Split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)    #split dataset into N folds
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)      #remove the current fold that we are looking at
        train_set = sum(train_set, [])      #add empty list AT THE END
        test_set = list()
        for row in fold:        #take numbers from row and put into test_set
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]      #actual values are the last values in each row(species)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

#Finding the euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

#Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup : tup[1])      #Sorts by second term in each tuple (lambda returns the value of the second term, it is necessary)
    neighbors = list()
    for i in range(num_neighbors):              #We get the top K values in a neighbors list
        neighbors.append(distances[i][0])
    return neighbors

#Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]          #creating a list of class values in neighbors (what type they are)
    prediction = max(set(output_values), key = output_values.count)     #which class is most near the given datapoint
    return prediction

#kNN algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions

#Testing kNN algo on any dataset
filename = 'INSERT_FILE_NAME_HERE'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
#convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
#evaluate algo, number of folds and neighbors is up to you
n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
# define a new record
row = [5.7,2.9,4.2,1.3]
# predict the label
label = predict_classification(dataset, row, num_neighbors)
print('Data=%s, Predicted: %s' % (row, label))
