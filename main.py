import numpy as np
from surprise import Reader, Dataset, KNNBaseline
 
reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(0,5))
data = Dataset.load_from_file('base.csv', reader=reader)
trainset = data.build_full_trainset()

#Train the algoritihm to compute the similarities between users
sim_options = {'name': 'pearson_baseline'}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

#First User ID
user_test_id = 413

user_test_neighbors = algo.get_neighbors(user_test_id, k=5)

print('The 5 nearest neighbors of User Test are:')
for user_id in user_test_neighbors:
    print(user_id)


