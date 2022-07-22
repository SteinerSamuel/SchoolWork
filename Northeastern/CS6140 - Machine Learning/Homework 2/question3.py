import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

d1 = pd.read_csv('data/airfoil_self_noise.csv')
d2 = pd.read_csv('data/Concrete_Data.csv')
d3 = pd.read_csv('data/winequality-red.csv')


def SED_objective(X, y, W):
    # part 1
    X = X.T
    W = W.reshape(W.shape[0], 1)
    y = y.reshape(1, y.shape[0])

    distance = np.subtract(W.T.dot(X), y)
    norm = np.linalg.norm(W)
    part_1 = distance/norm

    #part_2
    value_1 = (1/norm * X).transpose()
    value_2 = (1/norm**3) * np.subtract(X.T.dot(W), y.T).dot(W.T)


    part_2 = value_1 - value_2


    sum_delta = part_1.dot(part_2)
    return (2* sum_delta).flatten()

def SSE_objective(X, y, W):
    predictions = X.dot(W) # \sigma {X_ij * W_j} 
    errors = np.subtract(y, predictions) # y_i - sigma 
        
    sum_delta =  X.transpose().dot(errors * 2)
    return -sum_delta

def gradient_descent(X, y, W, alpha, tollerance, limit, objective):
    convergance = False
    step = 0
    while not convergance and step < limit:
        # input("pause")

        W_old = W
        W = W - alpha * objective(X, y, W)
        convergance = np.allclose(W_old, W, tollerance)
        step += 1

    return W, step

def run_gradient_on_data(df, alpha, objective, normalize):
    
    X = df.values[:, 0:-1]  # get input values from first two columns
    y = df.values[:, -1]  # get output values from last coulmn
  
    
    delta_limit = .00001
    if normalize:
        y = (y - np.min(y, axis = 0))/np.ptp(y, axis= 0)
        X = (X - np.min(X, axis = 0))/np.ptp(X, axis= 0)  # normalize data so it can be worked with easier
  
    X = np.hstack((np.ones((X.shape[0],1)), X)) # this adds X_i0 as 1 
    W = np.ones(X.shape[1])
    W_max_likelihood = np.linalg.solve(np.dot(X.T,X), np.dot(X.T, y))
    # print(W_max_likelihood)
    W_1, step = gradient_descent(X,y,W, alpha, delta_limit, 100000, objective)
    
    y_pred = X.dot(W_1)
    # print(y_pred.shape)
    # print(step)
    # print(W_1)

    W_offset = W_max_likelihood - W_1

    R2 = r2_score(y, y_pred)

    return W_max_likelihood, W_1, step, R2, W_offset


final_df = pd.DataFrame(columns=["name","W Guess", "W Max Likelihood", "R2", "W_offset", "step"])


returns = run_gradient_on_data(d1, 20*(10**-13), SSE_objective, False)
final_df.loc[len(final_df.index)] = ["Airfoil Self Noise SSE", returns[1], returns[0], returns[3], returns[4], returns[2]]
returns = run_gradient_on_data(d2, 325*(10**-13), SSE_objective, False)
final_df.loc[len(final_df.index)] = ["Concrete Data SSE ", returns[1], returns[0], returns[3], returns[4], returns[2]]
returns = run_gradient_on_data(d3, 1*(10**-9), SSE_objective, False)
final_df.loc[len(final_df.index)] = ["Wine Quality Red SSE ", returns[1], returns[0], returns[3], returns[4], returns[2]]

returns = run_gradient_on_data(d1, .0002, SSE_objective, True)
final_df.loc[len(final_df.index)] = ["Airfoil Self Noise SSE Normalized", returns[1], returns[0], returns[3], returns[4], returns[2]]
returns =run_gradient_on_data(d2, .0002, SSE_objective, True)
final_df.loc[len(final_df.index)] = ["Concrete Data SSE Normalized", returns[1], returns[0], returns[3], returns[4], returns[2]]
returns =  run_gradient_on_data(d3, .0002, SSE_objective, True)
final_df.loc[len(final_df.index)] = ["Wine Quality Red SSE Normalized", returns[1], returns[0], returns[3], returns[4], returns[2]]

returns = run_gradient_on_data(d1, 20*(10**-13), SED_objective, False)
final_df.loc[len(final_df.index)] = ["Airfoil Self Noise SED", returns[1], returns[0], returns[3], returns[4], returns[2]]
returns =  run_gradient_on_data(d2, 325*(10**-13), SED_objective, False)
final_df.loc[len(final_df.index)] = ["Concrete Data SED", returns[1], returns[0], returns[3], returns[4], returns[2]]
returns = run_gradient_on_data(d3,  1*(10**-9), SED_objective, False)
final_df.loc[len(final_df.index)] = ["Wine Quality Red SED", returns[1], returns[0], returns[3], returns[4], returns[2]]

returns =  run_gradient_on_data(d1, .000003 , SED_objective, True)
final_df.loc[len(final_df.index)] = ["Airfoil Self Noise SED Normalized", returns[1], returns[0], returns[3], returns[4], returns[2]]
returns = run_gradient_on_data(d2, .000003 , SED_objective, True)
final_df.loc[len(final_df.index)] = ["Concrete Data SED Normalized", returns[1], returns[0], returns[3], returns[4], returns[2]]
returns = run_gradient_on_data(d3, .000003 , SED_objective, True)
final_df.loc[len(final_df.index)] = ["Wine Quality Red SED Normalized", returns[1], returns[0], returns[3], returns[4], returns[2]]

final_df.to_csv("data/final_data.csv")