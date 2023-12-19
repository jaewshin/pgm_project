import numpy as np 
import matplotlib.pyplot as plt

def monte_carlo_attacks(generated_set, test_set, membership_idx):
    '''
    Compute the distance between the test set and the generated set and 
    use that as a probability of a data point in the test set used 
    in the training set.
    
    Input:
        train_set: list, the training set
        generated_set: list, the generated set
        test_set: list, the test set
        membership_idx: list, list indicating whether a data point in the test set belongs to the training set
        eps: float, the epsilon for the attack (hyperparameter)
    
    Output:
        probs: list, the probability of a data point in the test set 
              used in the training set
        acc: the accuracy of the attack
    '''
    dists = np.zeros((len(test_set), len(generated_set)))
    for i in range(len(test_set)):
        for j in range(len(generated_set)):
            dists[i, j] = np.count_nonzero(test_set[i] != generated_set[j])
    flat_distances = dists.flatten()
    eps = np.percentile(flat_distances, 0.1)

    probs = [] 

    for i in range(len(test_set)):
        test_point = test_set[i]
        indicator = 0 
        for j in range(len(generated_set)):
            generated_point = generated_set[j]
            distance = np.count_nonzero(test_point != generated_point)
            if distance < eps:
                indicator += 1
        prob = np.sum(indicator) / len(generated_set)
        probs.append(prob) 
    
    probs = np.array(probs)
    # putative_train = np.where(probs > 0)
    acc = 1 - np.count_nonzero(membership_idx[np.where(probs > 0)])/len(membership_idx[np.where(probs > 0)])
    return probs, acc
