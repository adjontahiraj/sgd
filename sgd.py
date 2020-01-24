# import numpy as np
# import random
# import csv
# import math

# ideal_alpha = 0;
# ideal_lambda = 0;
# ideal_model = np.zeros(8)

# #itialize weights with random values using a normal distribution with mean 0.04 and standard deviation 0.015
# def initial_values():
#     out = [float(random.gauss(0,0.015)) for x in range(7)]
#     #0 is out initial b
#     out.append(0);
#     return(np.array(out))

# #First we need to extract the data from the csv file so we can train and validate our model.
# def extract_data(file):
#     #features to hold the data and results to hold the actual chances of admit
#     features = [];
#     admission = []
#     count = 0;
#     #we need to get the features and divide them by the total scores you can get and hold them in out featurs vector
#     #and hold the chance of admit in the admission vector
#     with open(file, 'r') as csv_data:
#         r = csv.DictReader(csv_data)
#         for row in r:
#             #features holds the feature sets for each admission application
#             data = np.array([float(row['GRE Score'])/340, float(row['TOEFL Score'])/120, float(row['University Rating'])/5, float(row['SOP'])/5, float(row['LOR '])/5, float(row['CGPA'])/10, float(row['Research']),1]);
#             features.append(data)
#             #admission holds the chance of admission for each feature set with the same index in features vector
#             coa = float(row['Chance of Admit ']);
#             admission.append(coa);
#     return(features, admission)

# #x = numpy arr of size [n x k], n being num of samples, k being num of features
# #y = numpy arr of size [n x 1], n being num of training samples
# #alpha = learning rate (list of two floats [a_min, a_max])
# #lam = regularization weight (a list of two floats [lam_min, lam_max])
# #nepoch = max number of training epochs (an integer)
# #epsilon = error bound to stop iteration prematurely if e' is less than the given bound
# #params = numpy arr of size [(k+1) x 1], that represents the current paramater values W_0, ..., W_k-1,b
# def SGDSolver(x, y = None, alpha = None, lamb = None, nepoch = None, epsilon = None, params = None ):
#     if(y!= None):
#         #print("\n\nTraining: \n")
#         initial_model = initial_values()
#         initial_error = 100.0
#         lowest_error = initial_error
#         best_model = initial_model
#         alpha = [0.001, 0.01]
#         lamb = [0.0001, 0.001]
#         #alph = 0.05
#         #lam = 0.001
#         nepoch = 10
#         #we will loop through 10 different alphas and for each alpha loop through 10 different lambdas to compute
#         #2D grid search
#         for alph in np.arange(alpha[0], alpha[1], ((alpha[1]-alpha[0])/20)):
#             for lam in np.arange(lamb[0], lamb[1], ((lamb[1]-lamb[0])/10)):
#                 #looping through each epoch
#                 model = initial_model
#                 error = 0
#                 for ep in range(nepoch):
#                     #print("Epoch: " ,ep, " out of " ,nepoch, " " )
#                     for feature_set in range(0,(len(x)/20), 20):
#                         #print("Batch " ,(feature_set+1), " out of ",len(x)/10," ")
#                         #computing y^ with current model
#                         #y_hat = compute_y_hat(model, x[feature_set]);
#                         y_hat = np.zeros(20)
#                         for i in range(len(y_hat)):
#                             y_hat[i] = compute_y_hat(model, x[feature_set+i])
#                         #computing the loss with current model
#                             total_loss = 0
#                             total_loss += compute_loss(y_hat[i], y[feature_set +i], lam, model)
#                         total_loss = total_loss/20
#                         #loss = compute_loss(y_hat, y[feature_set], lam, model)
#                         #print("Loss: ",total_loss,"\n")
#                         #updating weights using SGD
#                         model = update_weights(model, x[feature_set:(feature_set+20)], y_hat, y[feature_set:(feature_set+20)], alph, lam)
#                 #return model;

#                 #print("\n\n Validation: \n")

#                 for feature_set in range(len(x)):
#                     predicted_out = compute_y_hat(model, x[feature_set])
#                     error += abs(predicted_out - y[feature_set])
#                 error = error/len(x)
#                 if error <= lowest_error:
#                     lowest_error = error;
#                     best_model = model;
#                     global ideal_model
#                     ideal_model = best_model
#                     global ideal_alpha 
#                     ideal_alpha= alph
#                     global ideal_lambda
#                     ideal_lambda = lam
#         return(lowest_error, ideal_model)
#     else:
#         #print("\n\nTesting: \n")
#         admission_out = np.zeros(len(x))
#         for i in range(len(x)):
#             admission_out[i] = params[-1]
#             for param in range(7):
#                 admission_out[i] += x[i][param] * params[param]
        
#         return(admission_out)



# #function to compute predicted output with given model
# def compute_y_hat(model, feature_set):
#     #setting it to b
#     out = model[-1]
#     for i in range(7):
#         out += model[i] * feature_set[i]
#     return out

# #function to compute loss function with given prediction, actual output, lambda, and current model weights.
# def compute_loss(y_hat, y, lam, model):
#     avg_weights = 0
#     for i in range(7):
#         avg_weights += model[i];
#     avg_weights = avg_weights/7;
#     loss = ((y_hat - y)*(y_hat - y)) + (lam * (avg_weights*avg_weights))
#     return loss

# #updating weights by calculating the gradient
# def update_weights(model, inputs, y_hat, y, learning_rate, regularization_weight):
#     avg_input = np.mean(inputs, axis=0)
#     new_model = np.zeros(8)
#     for i in range(len(model)-1):
#         new_model[i] = model[i] - (learning_rate*( (2*avg_input[i])*( np.average(y_hat) - np.average(y) ) + 2*regularization_weight*model[i]))
#     new_model[-1] = model[-1] - (learning_rate * ( (2*(np.average(y_hat) - np.average(y)) )) )
#     return new_model

# batch_size = 10
# output_info = []


# #for testing purposes
# in_features, in_admission = extract_data('Admission_predict.csv')
# test_features, test_admission = extract_data('test.csv')
# err, model = SGDSolver(in_features, in_admission)
# print("\n\n This is the best current model: \n")
# print(model)
# #print("\n\nThis is the ideal model: \n")
# #print(ideal_model)
# print("\n\n This is the error of the best current model for epoch 1: ")
# print(err)
# print("\n\n Alpha: ", ideal_alpha, " Lambda: ", ideal_lambda, "\n")


# test = SGDSolver(test_features, params = ideal_model)
# #print( "\nThese are the predicted admissions for test input: \n")
# #print(test)
# #print( "\nThese are the actual admissions for the test input: \n")

# #manual error calculation
# test_admission = np.asarray(test_admission, dtype=np.float32)
# print(test_admission)
# testing_error = 0
# for i in range(len(test)):
#     testing_error += abs(test[i] - test_admission[i])
# testing_error = testing_error/len(test)

# print( "\n\nError during test was: ", testing_error )





































import numpy as np
import random
import csv
import math

ideal_alpha = 0;
ideal_lambda = 0;
ideal_model = np.zeros(8)

#itialize weights with random values using a normal distribution with mean 0.04 and standard deviation 0.015
def initial_values():
    out = [float(random.gauss(0,0.015)) for x in range(7)]
    #0 is out initial b
    out.append(0);
    return(np.array(out))

#First we need to extract the data from the csv file so we can train and validate our model.
def extract_data(file):
    #features to hold the data and results to hold the actual chances of admit
    features = [];
    admission = []
    count = 0;
    #we need to get the features and divide them by the total scores you can get and hold them in out featurs vector
    #and hold the chance of admit in the admission vector
    with open(file, 'r') as csv_data:
        r = csv.DictReader(csv_data)
        for row in r:
            #features holds the feature sets for each admission application
            data = np.array([float(row['GRE Score'])/340, float(row['TOEFL Score'])/120, float(row['University Rating'])/5, float(row['SOP'])/5, float(row['LOR '])/5, float(row['CGPA'])/10, float(row['Research']),1]);
            features.append(data)
            #admission holds the chance of admission for each feature set with the same index in features vector
            coa = float(row['Chance of Admit ']);
            admission.append(coa);
    return(features, admission)

#x = numpy arr of size [n x k], n being num of samples, k being num of features
#y = numpy arr of size [n x 1], n being num of training samples
#alpha = learning rate (list of two floats [a_min, a_max])
#lam = regularization weight (a list of two floats [lam_min, lam_max])
#nepoch = max number of training epochs (an integer)
#epsilon = error bound to stop iteration prematurely if e' is less than the given bound
#params = numpy arr of size [(k+1) x 1], that represents the current paramater values W_0, ..., W_k-1,b
def SGDSolver(x, y = None, alpha = None, lamb = None, nepoch = None, epsilon = None, params = None ):
    if(y!= None):
        print("\n\nTraining: \n")
        initial_model = initial_values()
        initial_error = 100.0
        lowest_error = initial_error
        best_model = initial_model
        alpha = [0.001, 0.01]
        lamb = [0.0001, 0.0010]
        #alph = 0.05
        #lam = 0.001
        nepoch = 5
        #we will loop through 10 different alphas and for each alpha loop through 10 different lambdas to compute
        #2D grid search
        for alph in np.arange(alpha[0], alpha[1], ((alpha[1]-alpha[0])/20)):
            for lam in np.arange(lamb[0], lamb[1], ((lamb[1]-lamb[0])/10)):
                #looping through each epoch
                model = initial_model
                error = 0
                for ep in range(nepoch):
                    #print("Epoch: " ,ep, " out of " ,nepoch, " " )
                    for feature_set in range(len(x)):
                        #print("Batch " ,feature_set+1, " out of ",len(x)," ")
                        #computing y^ with current model
                        y_hat = compute_y_hat(model, x[feature_set]);
                        #computing the loss with current model
                        loss = compute_loss(y_hat, y[feature_set], lam, model)
                        #print("Loss: ",loss,"\n")
                        #updating weights using SGD
                        model = update_weights(model, x[feature_set], y_hat, y[feature_set], alph, lam)
                #return model;

                #print("\n\n Validation: \n")

                for feature_set in range(len(x)):
                    predicted_out = compute_y_hat(model, x[feature_set])
                    error += abs(predicted_out - y[feature_set])
                error = error/len(x)
                if error <= lowest_error:
                    lowest_error = error;
                    best_model = model;
                    global ideal_model
                    ideal_model = best_model
                    global ideal_alpha 
                    ideal_alpha= alph
                    global ideal_lambda
                    ideal_lambda = lam
        return(lowest_error, ideal_model)
    else:
        #print("\n\nTesting: \n")
        admission_out = np.zeros(len(x))
        for i in range(len(x)):
            admission_out[i] = params[-1]
            for param in range(7):
                admission_out[i] += x[i][param] * params[param]
        
        return(admission_out)



#function to compute predicted output with given model
def compute_y_hat(model, feature_set):
    #setting it to b
    out = model[-1]
    for i in range(7):
        out += model[i] * feature_set[i]
    return out

#function to compute loss function with given prediction, actual output, lambda, and current model weights.
def compute_loss(y_hat, y, lam, model):
    avg_weights = 0
    for i in range(7):
        avg_weights += model[i];
    avg_weights = avg_weights/7;
    loss = ((y_hat - y)*(y_hat - y)) + (lam * (avg_weights*avg_weights))
    return loss

#updating weights by calculating the gradient
def update_weights(model, inputs, y_hat, y, learning_rate, regularization_weight):
    new_model = np.zeros(8)
    for i in range(len(model)-1):
        new_model[i] = model[i] - (learning_rate*( (2*inputs[i]*(y_hat - y)) + 2*regularization_weight*model[i]))
    new_model[-1] = model[-1] - (learning_rate * ( (2*(y_hat - y))) )
    return new_model


#for testing purposes
in_features, in_admission = extract_data('Admission_predict.csv')
test_features, test_admission = extract_data('test.csv')
err, model = SGDSolver(in_features, in_admission)
# print("\n\n This is the best current model: \n")
# print(model)
# print("\n\nThis is the ideal model: \n")
# print(ideal_model)
print("\n\n This is the error of the best current model: ")
print(err)
print("\n\n Alpha: ", ideal_alpha, " Lambda: ", ideal_lambda, "\n")


test = SGDSolver(test_features, params = ideal_model)
# print( "\nThese are the predicted admissions for test input: \n")
# print(test)
# print( "\nThese are the actual admissions for the test input: \n")

#manual error calculation
test_admission = np.asarray(test_admission, dtype=np.float32)
# print(test_admission)
testing_error = 0
for i in range(len(test)):
    testing_error += abs(test[i] - test_admission[i])
testing_error = testing_error/len(test)

print( "\n\nError during test was: ", testing_error )
