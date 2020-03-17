import pandas as pd
import numpy as np

#defining the learning rate is one of the most difficult things in machine learning and neural networks
#but this one was not that hard, some basic tests were enough
n = 0.2

#function to standardize the data via z-score
def standardize_df(df):
    
    for col in df.columns:
        datatype = df[col].dtypes

        if (datatype == 'float64') or (datatype == 'int64'):
            std = df[col].std()
            mean = df[col].mean()
            df[col] = (df[col] - mean) / std

    return df

#this update function changes the perceptron weights based on a mistaken classification
def update_weights(weights, d, point):
    point[len(point)-1] = 1
    
    #for each weight wi, we are doing wi' = wi + learning_rate * expected_output * xi to find a new weight
    for i in range(len(weights)):
        weights[i] = n*d*point[i]
       
    return weights

#simple function to predict the outcome using the perceptron weights
def predict_outcome(weights, point):
    point[len(point)-1] = 1
    result = np.dot(weights, point)
    
    if result > 0:
        return 1
    else:
        return -1

#despite the fancy name, this function is basically just optimizing the perceptron weights, that is, applying
#the core logical basis of the algorithm
def find_best_hyperplane(weights, repetitions, df):
    
    #the number of repetitions is usually another tough thing to determine, in this case 15 showed to be
    #just the right quantity
    for k in range(repetitions):
        
        #for each point in the training set we try to predict its output using the weights we have
        for i, row in df.iterrows():
            outcome = row['name']
            point = row.values
            predicted_outcome = predict_outcome(weights, point)
            
            #if we miss the prediction we call the update method
            if predicted_outcome != outcome:
                weights = update_weights(weights, outcome, point)
    
    return weights
         
#importing the dataframe, standardizing its values and diving traindf and testdf        
df = pd.read_csv("http://abre.ai/orgvsgrpf")
df = standardize_df(df)
df['name'] = df['name'].replace({'orange': 1, 'grapefruit': -1})
traindf = pd.concat([df.iloc[:4000], df.iloc[5000:9000]]).sample(frac = 1).reset_index(drop = True)
testdf = pd.concat([df.iloc[4000:5000], df.iloc[9000:]]).sample(frac = 1).reset_index(drop = True)

#ws are the weights to the perceptron, the hyperplane equation coeficients in a more mathematical view
ws = [0]*len(df.columns)
ws = find_best_hyperplane(ws, 15, traindf)

#just evaluating our implementation, with this configuration of learning rate and quantity of repetitions we are
#getting results that usually vary between 90 and 100% accuracy.
correct = 0 
wrong = 0

for i in range(len(testdf.index)):
    algorithm_outcome = predict_outcome(ws, testdf.iloc[i].values)
    real_outcome = testdf.iloc[i]['name']
    
    if(algorithm_outcome == real_outcome):
        correct += 1
    else:
        wrong += 1

        
print('Total: '+str(correct+wrong))
print('Correct: '+str(correct))
print('Wrong: '+str(wrong)+'\n')
print('Accuracy: '+str(round(100*(correct/(correct+wrong))))+'%')