# -*- coding: utf-8 -*-

from config_glove import Config
from graph_glove import Graph
from model.sdne_glove import SDNE
from utils.utils import *
import scipy.io as sio
import time
import copy
from optparse import OptionParser
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = OptionParser()
    parser.add_option("-c",dest = "config_file",default="config/airports.ini", action = "store", metavar = "CONFIG FILE")
    options, _ = parser.parse_args()
    if options.config_file == None:
        raise IOError("no config file specified")

    config = Config(options.config_file)
    # print(config.dbn_init)

    train_graph_data = Graph(config.train_graph_file)
   
    if config.origin_graph_file:
        origin_graph_data = Graph(config.origin_graph_file)

    if config.label_file:
        train_graph_data.load_label_data(config.label_file)
    
    config.struct[0] = train_graph_data.N

    model = SDNE(config)
    model.do_variables_init(train_graph_data)
    print("\n", end="")
    embedding = None
    while (True):
        mini_batch = train_graph_data.sample(config.batch_size, do_shuffle = False)
        if embedding is None:
            embedding = model.get_embedding(mini_batch)
        else:
            embedding = np.vstack((embedding, model.get_embedding(mini_batch))) 
        if train_graph_data.is_epoch_end:
            break
    
    epochs = 0
    batch_n = 0
    
    time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    tt = time.strftime("%Y-%m-%d_%H-%M-%S") #time.ctime() #.replace(' ','-')
    path = ".\\result\\" + config.embedding_filename + '_' + tt
    os.system("mkdir " + path)
    fout = open(path + "\\log.txt", "w")  
    model.save_model(path + '\\epoch0.model')

    sio.savemat(path + '\\embedding.mat', {'embedding': embedding})
    while (True):
        if train_graph_data.is_epoch_end:
            loss = 0
            if epochs % config.display == 0:
                embedding = None
                while (True):
                    mini_batch = train_graph_data.sample(config.batch_size, do_shuffle = False)
                    loss += model.get_loss(mini_batch)
                    if embedding is None:
                        embedding = model.get_embedding(mini_batch)
                    else:
                        embedding = np.vstack((embedding, model.get_embedding(mini_batch))) 
                    if train_graph_data.is_epoch_end:
                        break

                print("Epoch : %d loss : %.3f\n" % (epochs, loss))
                print("Epoch : %d loss : %.3f" % (epochs, loss), file=fout)
                #if config.check_reconstruction:
                    #print(epochs, "reconstruction:", check_reconstruction(embedding, origin_graph_data, config.check_reconstruction), file=fout)
                #if config.check_link_prediction:
                    #print(epochs, "link_prediction:", check_link_prediction(embedding, train_graph_data, origin_graph_data, config.check_link_prediction), file=fout)
                t1 = time.time()

                if config.check_link_prediction:
                    print(epochs, 'link_prediction - AUC: ', check_link_prediction_auc(embedding, train_graph_data, origin_graph_data), file=fout)
                    t2 = time.time()
                    print('time cost: ', t2-t1, file=fout)

                #if config.check_classification:
                    #data = train_graph_data.sample(train_graph_data.N, do_shuffle = False, with_label = True)
                    #print(epochs, "classification", check_multi_label_classification(embedding, data.label), file=fout)
                #if config.check_classification:
                #    data = train_graph_data.sample(train_graph_data.N, do_shuffle = False, with_label = True)
                #    idx=range(train_graph_data.N)
                #    X=embedding[idx, :]
                #    y=data.label[idx]

                #    t1=time.time()
                #    x=t_sne(X,n=2,max_iter=500,neighbors=20)
                #    t2=time.time()

                #   plt.figure(figsize=(13,10))
                #    plt.scatter(x[:,0], x[:,1], c=y, cmap='jet')
                #    plt.axis('off')
                #    plt.savefig(path + '\\' + str(epochs) + 'enco-ve.png')
                #    plt.show()
                #print('time: ', t2-t1)
                fout.flush()
                model.save_model(path + '\\epoch' + str(epochs) + ".model")
            if epochs == config.epochs_limit:
                print("exceed epochs limit terminating")
                break
            epochs += 1
        mini_batch = train_graph_data.sample(config.batch_size)
        loss = model.fit(mini_batch)

    sio.savemat(path + '/embedding.mat',{'embedding':embedding})
    fout.close()
