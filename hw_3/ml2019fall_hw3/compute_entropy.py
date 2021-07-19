import numpy as np

def entropy(data):
    _sum = np.sum(data)
    prob = data/_sum
    return -np.dot(prob, np.log2(prob).T)

if __name__ == "__main__":
    data = np.array([10, 95, 5, 90, 80, 20, 120, 30])
    
    root = entropy(np.array([np.sum(data[:4]), np.sum(data[4:])]))
    # root
    gender_1 = (205/450)*entropy(np.array([105, 100])) + (245/450)*entropy(np.array([95, 150]))
    gpa_1 = (215/450)*entropy(np.array([15, 200])) + (235/450)*entropy(np.array([185, 50]))
    print(root, gpa_1)
    print(root-gpa_1)
    # gpa is better
    #-------------------------------------------------------------------------
    low_en = entropy(np.array([15, 200]))
    low_gender_2 = (90/215)*entropy(np.array([10, 80])) + (125/215)*entropy(np.array([5, 120]))
    print(low_en-low_gender_2)
    high_en = entropy(np.array([50, 185]))
    high_gender_2 = (115/235)*entropy(np.array([95, 20])) + (120/235)*entropy(np.array([90, 30]))
    print(high_en-high_gender_2)
