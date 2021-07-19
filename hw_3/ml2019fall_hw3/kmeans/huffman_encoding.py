import numpy as np

def huffman_ecoding(idx):
    _, w = np.unique(idx, return_counts=True)
    size = 0

    while len(w) >=2:
        fsi = np.argmin(w)
        fs = w[fsi]
        w = np.delete(w, fsi)
        ssi = np.argmin(w)
        ss = w[ssi]
        w = np.delete(w, ssi)
        w = np.append(w, fs+ss)
        size += fs+ss
    return size

if __name__ == "__main__":
    huffman_ecoding(np.array([1,2,3,3,4,4,4,5,5,5,5]))
