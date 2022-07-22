import random
import numpy as np
import pandas as pd

random.seed(234503245)

def solve(bl, tr, p) :
    truth = [p[0] > bl[0], p[0] < tr[0], p[1] > bl[1], p[1] < tr[1]]
    if (all(truth)) :
        return True
    else:
        return False

# size = [250, 1000, 10000]
size = [10000]

x_1_vals = [-6,6]
x_2_vals = [-4,4]
positives = [[(-4,0), (-1,3)], [(-2,-4),(1,-1)], [(2,-2), (5,1)]]

for s_ in size:
    print(s_)
    X = {}
    X['x_1'] = 12 * np.random.random_sample(size=size) - 6
    X['x_2'] = 8 * np.random.random_sample(size=size) - 4

    # X = zip(x_1, x_2)
    # 


    df = pd.DataFrame(data=X)
 
    df['y'] = df.apply( lambda row: 1 if any([solve(positive[0], positive[1], [row.x_1, row.x_2]) for positive in positives]) else 0, axis=1)
    df.to_csv(f'n_{s_}.csv', index=False)


    # for x in X:
    #     s = [solve(positive[0], positive[1], x) for positive in positives]
    #     if any(s):
    #         df.loc[len(df.index)] = [x[0], x[1], 1]
    #     else:
    #         df.loc[len(df.index)] = [x[0], x[1], 0]


    