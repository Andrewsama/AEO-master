import scipy.io as sio
import numpy as np
from utils.utils import *
import random
import copy
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix

class Graph(object):
    def __init__(self, file_path):
        #self.config = config
        suffix = file_path.split('.')[-1]
        self.st = 0
        self.is_epoch_end = False
        matrix_dict = sio.loadmat('GraphData\\airports_ppmi_4_1.mat')
        self.COMATRIX = matrix_dict.get('pp_airports')
        self.N, _ = self.COMATRIX.shape

        if suffix == "txt":
            fin = open(file_path, "r")
            firstLine = fin.readline().strip().split()
            self.N = int(firstLine[0])
            self.E = int(firstLine[1])
            self.__is_epoch_end = False
            self.adj_matrix = dok_matrix((self.N, self.N), np.int_)
            count = 0
            for line in fin.readlines():
                line = line.strip().split()
                x = int(line[0])
                y = int(line[1]) # for index from 1 to N
                z = int(line[2])
                self.adj_matrix[x, y] += z
                self.adj_matrix[y, x] += z
                count += 1
            fin.close()
            self.adj_matrix = self.adj_matrix.tocsr()

        self.order = np.arange(self.N)
        print("Vertexes : %d\n\n" % (self.N))

    def get_cooccurrence(self, matrix, seq_len, alpha):
        matrix = matrix - np.diag(np.diag(matrix))
        D = np.diag(np.reciprocal(np.sum(matrix, axis=0)))
        matrix = np.dot(D, matrix)

        P0 = np.eye(self.N, dtype='float32')
        COMATRIX = np.zeros((self.N, self.N), dtype='float32')
        P = np.eye(self.N, dtype='float32')
        for index in range(0, seq_len):
            P = alpha * np.dot(P, matrix) + P0 * (1 - alpha)
            COMATRIX += P

        return COMATRIX

    def load_label_data(self, filename):
        with open(filename, 'r') as f:
            fL = f.readline().strip().split()
            self.label = np.zeros([self.N], np.int)
            lines = f.readlines()
            index = 0
            for line in lines:
                self.label[index] = int(line)
                index += 1

    def sample(self, batch_size, do_shuffle = True, with_label = False):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self.order[0:self.N])
            else:
                self.order = np.sort(self.order)
            self.st = 0
            self.is_epoch_end = False 
        mini_batch = Dotdict()
        en = min(self.N , self.st + batch_size)
        index = self.order[self.st:en]

        mini_batch.COMATRIX = self.COMATRIX[index, :]

        if not with_label:
            index1 = random.randint(0, self.N-batch_size-1)
            mini_batch.adjacent_matriX = self.COMATRIX[index, index1:index1+batch_size]
        if with_label:
            mini_batch.label = self.label[index]
        if (en == self.N):
            en = 0
            self.is_epoch_end = True
        self.st = en
        return mini_batch