import numpy as np

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    J: index of item with large probability at each position
    q: probability of the smaller item at each position
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(probs, sample_num):
    J, q = alias_setup(probs)
    K = len(J)
    assert sample_num <= K
    sample_data = set()
    while len(sample_data) < sample_num:
        kk = int(np.floor(np.random.rand() * K))
        item = kk if np.random.rand() < q[kk] else J[kk]
        if item not in sample_data:
            sample_data.add(item)
    return sample_data

def sample_data():
    ...
