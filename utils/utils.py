import numpy
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import pdb
from sklearn.manifold import TSNE

from sklearn import metrics
import copy

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def cal_matrix_P(X,neighbors):
    entropy=numpy.log(neighbors)
    n1,n2=X.shape
    D=numpy.square(metrics.pairwise_distances(X))
    D_sort=numpy.argsort(D,axis=1)
    P=numpy.zeros((n1,n1))

    for i in range(n1):
        Di=D[i,D_sort[i,1:]]
        P[i,D_sort[i,1:]]=cal_p(Di,entropy=entropy)
    P=(P+numpy.transpose(P))/(2*n1)
    P=numpy.maximum(P,1e-100)
    return P

def cal_p(D,entropy,K=50):
    beta=1.0
    H=cal_entropy(D,beta)
    error=H-entropy
    k=0
    betamin=-numpy.inf
    betamax=numpy.inf

    while numpy.abs(error)>1e-4 and k<=K:
        if error > 0:
            betamin=copy.deepcopy(beta)
            if betamax==numpy.inf:
                beta=beta*2
            else:
                beta=(beta+betamax)/2
        else:
            betamax=copy.deepcopy(beta)
            if betamin==-numpy.inf:
                beta=beta/2
            else:
                beta=(beta+betamin)/2
        H=cal_entropy(D,beta)
        error=H-entropy
        k+=1
    P=numpy.exp(-D*beta)
    P=P/numpy.sum(P)
    return P

def cal_entropy(D,beta):
    P=numpy.exp(-D*beta)
    sumP=sum(P)
    sumP=numpy.maximum(sumP,1e-200)
    H=numpy.log(sumP) + beta * numpy.sum(D * P) / sumP
    return H

def cal_matrix_Q(Y):
    n1,n2=Y.shape
    D=numpy.square(metrics.pairwise_distances(Y))
    Q=(1/(1+D))/(numpy.sum(1/(1+D))-n1)
    Q=Q/(numpy.sum(Q)-numpy.sum(Q[range(n1),range(n1)]))
    Q[range(n1),range(n1)]=0
    Q=numpy.maximum(Q,1e-100)
    return Q

def cal_gradients(P,Q,Y):
    n1,n2=Y.shape
    DC=numpy.zeros((n1,n2))

    for i in range(n1):
        E=(1+numpy.sum((Y[i,:]-Y)**2,axis=1))**(-1)
        F=Y[i,:]-Y
        G=(P[i,:]-Q[i,:])
        E=E.reshape((-1,1))
        G=G.reshape((-1,1))
        G=numpy.tile(G,(1,n2))
        E=numpy.tile(E,(1,n2))
        DC[i,:]=numpy.sum(4*G*E*F,axis=0)
    return DC

def cal_loss(P,Q):
    C=numpy.sum(P * numpy.log(P / Q))
    return C

def t_sne(X,n=2,neighbors=30,max_iter=200):
    data=[]
    n1,n2=X.shape
    P=cal_matrix_P(X,neighbors)
    Y=numpy.random.randn(n1,n)*1e-4
    Q = cal_matrix_Q(Y)
    DY = cal_gradients(P, Q, Y)
    A=200.0
    B=0.1
    for i in range(max_iter):
        data.append(Y)
        if i==0:
            Y=Y-A*DY
            Y1=Y
            error1=cal_loss(P,Q)
        elif i==1:
            Y=Y-A*DY
            Y2=Y
            error2=cal_loss(P,Q)
        else:
            YY=Y-A*DY+B*(Y2-Y1)
            QQ = cal_matrix_Q(YY)
            error=cal_loss(P,QQ)
            if error>error2:
                A=A*0.7
                continue
            elif (error-error2)>(error2-error1):
                A=A*1.2
            Y=YY
            error1=error2
            error2=error
            Q = QQ
            DY = cal_gradients(P, Q, Y)
            Y1=Y2
            Y2=Y
        if cal_loss(P,Q)<1e-3:
            return Y
        if numpy.fmod(i+1,10)==0:
            print('%s iterations the error is %s, A is %s'%(str(i+1),str(round(cal_loss(P,Q),2)),str(round(A,3))))
    return Y

def check_link_prediction_auc(embedding, train_graph_data, origin_graph_data):
    a = getSimilarity(embedding)
    N = origin_graph_data.N
    count = 0
    count1 = 0
    count2 = 0
    MAX_COUNT = 9604
    AUC = []
    for i in range(N):
        for j in range(N):
            x = i
            y = j
            if (x==y or train_graph_data.adj_matrix[x].toarray()[0][y] > 0):
                continue
            if (origin_graph_data.adj_matrix[x].toarray()[0][y] > 0):
                s = a[x][y]
                while (origin_graph_data.adj_matrix[x].toarray()[0][y] > 0):
                    x = np.random.randint(0, N)
                    y = np.random.randint(0, N)
                ss = a[x][y]
                count += 1
                if s > ss:
                    count1 += 1
                elif s == ss:
                    count2 += 1
            if count > MAX_COUNT:
                break

    AUC.append((1.0 * (count1 + 0.5*count2) / count))
    return AUC

def getSimilarity(result):
    print("getting similarity...")
    return np.dot(result, result.T)
    
def check_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        count_i = 0
        precisionK = []
        AP = []
        MAP = []
        sortedInd = sortedInd[::-1]
        for i in sortedInd:
            api = 0
            for ind in sortedInd:
                x = int(ind / data.N)
                y = ind % data.N
                count += 1
                if (data.adj_matrix[x].toarray()[0][y] == 1 or x == y):
                    cur += 1
                    api += (1.0 * cur / count) * 1
                precisionK.append(1.0 * cur / count)
                if count > max_index:
                    break
            AP.append(api)
            count_i += 1
            if count_i > max_index:
                break
        MAP.append(1.0 * sum(AP) / count_i)
        return precisionK, MAP

    precisionK, MAP = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    print("MAP\t", MAP)
    print("\n\n")
    ret.append(MAP)
    return ret

def check_link_prediction(embedding, train_graph_data, origin_graph_data, check_index):
    def get_precisionK(embedding, train_graph_data, origin_graph_data, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        #print(similarity)
        #print("\n")
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        count_i = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        N = train_graph_data.N
        AP = []
        MAP = []
        for i in sortedInd:
            api = 0
            for ind in sortedInd:
                x = int(ind / N)
                y = ind % N
                if (x == y or train_graph_data.adj_matrix[x].toarray()[0][y] == 1):
                    continue 
                count += 1
                if (origin_graph_data.adj_matrix[x].toarray()[0][y] == 1):
                    cur += 1
                    api += (1.0 * cur / count) * 1
                precisionK.append(1.0 * cur / count)
                if count > max_index:
                    break
            AP.append(api)
            count_i += 1
            if count_i > max_index:
                break
        MAP.append(1.0 * sum(AP) / count_i)
        return precisionK, MAP
    precisionK, MAP = get_precisionK(embedding, train_graph_data, origin_graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    
    print("MAP\t", MAP)
    print("\n\n")
    ret.append(MAP)
    return ret
 

def check_multi_label_classification(X, Y, test_ratio = 0.1):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape,np.bool)
        sort_index = np.flip(np.argsort(y_pred, axis = 1), 1)
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    
    ## small trick : we assume that we know how many label to predict
    y_pred = small_trick(y_test, y_pred)
    
    micro = f1_score(y_test, y_pred, average = "micro")
    macro = f1_score(y_test, y_pred, average = "macro")
    return("micro_f1: %.4f macro_f1 : %.4f" % (micro, macro))