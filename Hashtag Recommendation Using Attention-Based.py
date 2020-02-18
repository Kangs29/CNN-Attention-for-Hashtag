
# coding: utf-8

# In[ ]:

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')

import sys, os
import numpy as np
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def DataFile():
    with open("/content/drive/My Drive/vocabulary_keras_h.pkl", "rb") as f:
        data = pickle.load(f)
    vocabulary = data[0]
    hashtagVoc = data[2]
    vocabulary_inv = {}
    hashtagVoc_inv = {}
    vocabulary["<novocab>"] = 0

    for i in vocabulary.keys():
        vocabulary_inv[vocabulary[i]] = i
    for i in hashtagVoc.keys():
        hashtagVoc_inv[hashtagVoc[i]] = i

    val_data = []
    with open("/content/drive/My Drive/val_tlh_keras_h.bin", "rb") as f:
        val_data.extend(pickle.load(f))

    test_data = []
    with open("/content/drive/My Drive/test_tlh_keras_h.bin", "rb") as f:
        test_data.extend(pickle.load(f))

    train_data = []
    with open("/content/drive/My Drive/train_tlh_keras_h.bin", "rb") as f:
        train_data.extend(pickle.load(f))

    word_dict=dict()
    word_dict["#PAD"]=0
    word_dict["#BEGIN"]=1
    word_dict["#END"]=2
    
    for idx in vocabulary_inv:
        if vocabulary_inv[idx] == '<novocab>':
            continue
        else:
            word_dict[vocabulary_inv[idx]]=len(word_dict)

    return train_data, val_data, test_data, word_dict, hashtagVoc_inv


def DataProceesing(train_,Unique=False,RandomChoice=False):
    train_edit=[]
    for i in range(len(train_[0])):
        train_edit.append([[j for j in train_[0][i]],train_[2][i]])


    if Unique==True:
        trainText=dict()
        trainLabel=dict()

        numbering=0

        for num in range(len(train_edit)-2):
            testing = train_edit[num:num+2]
            if str(testing[0][0]+testing[0][1])==str(testing[1][0]+testing[1][1]):
                continue
            else:
                trainText[numbering] = testing[0][0]
                trainLabel[numbering] = testing[0][1]
                numbering+=1


        if str(testing[0][0]+testing[0][1]) != str(testing[1][0]+testing[1][1]):
                trainText[numbering] = testing[1][0]
                trainLabel[numbering] = testing[1][1]
                
        if RandomChoice==True:
            ran=np.random.choice([i for i in range(len(trainText))],len(trainText),replace=False)
        else:
            ran=[i for i in range(len(trainText))]
        
        train_data=[]
        trainT=[]
        trainL=[]

        for i in ran:
            trainT.append(trainText[i])
            trainL.append(trainLabel[i])

        train_data.append(np.array(trainT))
        train_data.append(np.array(trainL))
        train_data.append(np.array(trainL))

    else:
        if RandomChoice==True:
            ran=np.random.choice([i for i in range(len(train_edit))],len(train_edit),replace=False)
        else:
            ran=[i for i in range(len(train_edit))]

        train_data=[]
        trainT=[]
        trainL=[]
        
        for i in ran:
            trainT.append(train_edit[i][0])
            trainL.append(train_edit[i][1])

        train_data.append(np.array(trainT))
        train_data.append(np.array(trainL))
        train_data.append(np.array(trainL))

    print("Unique Sentence :",Unique)
    print("One Shuffle whether or not :",RandomChoice)
    return train_data


def AllPairData(train_data_):
    train_data=[]
    for num in range(len(train_data_[0])):
        sentence = [i for i in train_data_[0][num]]
        labels = train_data_[2][num]
        for label in labels:
            train_data.append([sentence,label])

    return train_data





class CNN_Att(nn.Module):
    def __init__(self,vocabulary_size, hashtag_size,window=5, comparison_rate=0.8,embedding_size=100,local_output=100,max_length=411,               num_filters=100,filter_sizes=[1,2,3],global_output=100):
        super(CNN_Att,self).__init__()
        
        #local hyperparameter
        self.window = window
        self.comparison_rate = comparison_rate
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.local_output = local_output
        self.max_length = max_length

        self.embedding=nn.Embedding(self.vocabulary_size,self.embedding_size)
        torch.nn.init.uniform_(self.embedding.weight,-0.01,0.01)

        self.att=nn.Linear(self.embedding_size,self.window)
        torch.nn.init.uniform_(self.att.weight,-0.01,0.01)
        self.att2=nn.Linear(self.embedding_size,self.local_output)
        torch.nn.init.uniform_(self.att2.weight,-0.01,0.01)

        #global hyperparameter
        self.num_filters=num_filters
        self.filter_sizes=filter_sizes
        self.global_output=global_output

        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters,[filter_size, self.embedding_size],padding=(0,0))                                     for filter_size in self.filter_sizes])
        torch.nn.init.uniform_(self.convs[0].weight,-0.01,0.01)
        torch.nn.init.uniform_(self.convs[1].weight,-0.01,0.01)
        torch.nn.init.uniform_(self.convs[2].weight,-0.01,0.01)

        self.multi_filter = nn.Linear(self.num_filters*len(self.convs),self.global_output)

        # combine
        self.hashtag_size=hashtag_size

        self.fin = nn.Linear(200,400)
        torch.nn.init.uniform_(self.fin.weight,-0.01,0.01)
        self.fin2 = nn.Linear(400,2987)
        torch.nn.init.uniform_(self.fin2.weight,-0.01,0.01)

    def forward(self,l_train_text,g_train_text):
        # local
        batch_size=l_train_text.size()[0]
        target_words=torch.zeros(batch_size,self.embedding_size).to(device)

        for pos in range(self.max_length):
            embed = self.embedding(l_train_text[:,pos:pos+self.window])  # [batch,window_word,embedding]
            attention = torch.tanh(self.att(embed))                      # [batch,window_word,window_score]
            score = torch.sum(attention,2)                               # [batch,window_word]


            comparison = self.comparison_rate*torch.min(score) + self.comparison_rate*torch.max(score) # 0.8*min + 0.2*max == scalar
            target_word = embed[:,int(self.window/2)]

            judge=torch.sum(target_word,1)>comparison                    # [batch]
            judge=judge.view(batch_size,1).repeat(1,self.embedding_size) # [batch,embedding]
            target_word=judge*target_word                                # trigger words or not
            target_words+=target_word                                    # folding
            target_words=target_words/self.max_length
            local_units=self.att2(target_words)                          # [batch,local_output]
            
        # global
        embed=self.embedding(g_train_text)
        embed=torch.unsqueeze(embed,1)

        pooling_output=[]
        for conv in self.convs:
            relu=F.relu(conv(embed))
            relu=torch.squeeze(relu,-1)
            pool=F.max_pool1d(relu,relu.size(2))
            pooling_output.append(pool)


        pooling_output=torch.squeeze(torch.cat(pooling_output,1),-1)     # [batch, num_filter*3]
        global_units = self.multi_filter(pooling_output)


        combine=torch.cat([local_units,global_units],1)                  # [batch,local+global]
        logit= F.relu(self.fin(combine))
        logit = self.fin2(logit)
        
        return logit


def PickData(train_data,train_nu,batch_size):
    batch_data = []
    for lines in train_data[train_nu:train_nu+batch_size]:
        sentence = lines[0]
        label=lines[1]
        batch_data.append([sentence,label])

    return batch_data


def Training(train_data,val_data,test_data,word_dict,hashtag_dict,
            max_length=411,epoch=30,batch_size=200,printing=1000):

    print(" - Data information")
    print("The number of training data   :",len(train_data))
    print("The number of validation data :",len(val_data[0]))
    print("The number of test data       :",len(test_data[0]))
    print("Vocabulary Size :",len(word_dict))
    print("Hashtag Size    :",len(hashtag_dict))
    
    
    vocabulary_size = len(word_dict)
    hashtag_size = len(hashtag_dict)

    model = CNN_Att(vocabulary_size,hashtag_size)
    model.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    loss_function = nn.CrossEntropyLoss()                               # loss function : Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)      # optimizer :adam
 
    now_epoch=0
    train_nu=0
    for iters in range(int(len(train_data)*20*epoch)):
        model.train()

        if train_nu >= len(train_data):
            now_epoch+=1
            print(" - Epoch is < %d > :" % now_epoch)
            train_nu=0
            random.shuffle(train_data)

        batch_train = PickData(train_data,train_nu,batch_size)
        batch_text,batch_y = zip(*batch_train)
        
        local_text=[]
        for lines in batch_text:
            local_text.append([1]*2+[int(i) for i in lines]+[2]*2)

        l_train_text = torch.LongTensor(local_text).to(device)
        g_train_text = torch.LongTensor(batch_text).to(device)
        train_label = torch.LongTensor(batch_y).to(device)

        logits=model(l_train_text,g_train_text)
        losses = loss_function(logits, train_label)
        optimizer.zero_grad()                                           # gradient to zero
        losses.backward()                                               # load backward function

        optimizer.step()                                                # update parameters

        train_nu+=batch_size
    
        if iters % printing == 0:
            V_precision=0
            nu=0
            for _ in range(int(len(val_data[0])/1000)+1):
                batch_text, batch_y = val_data[0][nu*1000:(nu+1)*1000], val_data[2][nu*1000:(nu+1)*1000]


                local_text=[]
                for lines in batch_text:
                    local_text.append([1]*2+[int(i) for i in lines]+[2]*2)


                l_train_text = torch.LongTensor(local_text).to(device)
                g_train_text = torch.LongTensor(batch_text).to(device)
                logits=model(l_train_text,g_train_text)


                val_prec = torch.argsort(-logits,1)
                if nu<int(len(val_data[0])/1000):
                    V_precision += sum([True for i in range(1000) if int(val_prec[i][0]) in val_data[2][nu*1000:(nu+1)*1000][i]])
                else:
                    V_precision += sum([True for i in range(len(batch_text)) if int(val_prec[i][0]) in val_data[2][nu*len(batch_text):(nu+1)*len(batch_text)][i]])

                nu+=1
            print("Iteration : %d, Valid_Precision : %.08f" % (iters,V_precision/len(val_data[0])))
            print(val_prec)

            Test_precision=0
            nu=0
            for _ in range(int(len(test_data[0])/1000)+1):
                batch_text, batch_y = test_data[0][nu*1000:(nu+1)*1000], test_data[2][nu*1000:(nu+1)*1000]


                local_text=[]
                for lines in batch_text:
                    local_text.append([1]*2+[int(i) for i in lines]+[2]*2)


                l_train_text = torch.LongTensor(local_text).to(device)
                g_train_text = torch.LongTensor(batch_text).to(device)
                logits=model(l_train_text,g_train_text)


                test_prec = torch.argsort(-logits,1)
                if nu<int(len(val_data[0])/1000):
                    Test_precision += sum([True for i in range(1000) if int(test_prec[i][0]) in test_data[2][nu*1000:(nu+1)*1000][i]])
                else:
                    Test_precision += sum([True for i in range(len(batch_text)) if int(test_prec[i][0]) in test_data[2][nu*len(batch_text):(nu+1)*len(batch_text)][i]])

                nu+=1

            print("Test_Precision : %.05f" % (Test_precision/len(test_data[0])))


# In[ ]:

Unique=True
RandomChoice=True
train_,val_data,test_data,word_dict,hashtag_dict=DataFile()
train_data_ = DataProceesing(train_,Unique=Unique,RandomChoice=RandomChoice)
train_data = AllPairData(train_data_)
Training(train_data,val_data,test_data,word_dict,hashtag_dict,printing=100)


# In[ ]:

#torch.save(save_model.state_dict(),"/content/drive/My Drive/CNN_ATT.pth")
model=CNN_Att(vocabulary_size,hashtag_size)
model.to(device)

model.load_state_dict(torch.load("./CNN_ATT.pth"))


# In[ ]:

for_do=[]
Test_precision=0
nu=0
for _ in range(int(len(test_data[0])/1000)+1):
    batch_text, batch_y = test_data[0][nu*1000:(nu+1)*1000], test_data[2][nu*1000:(nu+1)*1000]


    local_text=[]
    for lines in batch_text:
        local_text.append([1]*2+[int(i) for i in lines]+[2]*2)


    l_train_text = torch.LongTensor(local_text).to(device)
    g_train_text = torch.LongTensor(batch_text).to(device)
    logits=new(l_train_text,g_train_text)


    test_prec = torch.argsort(-logits,1)
    if nu<int(len(val_data[0])/1000):
        Test_precision += sum([True for i in range(1000) if int(test_prec[i][0]) in test_data[2][nu*1000:(nu+1)*1000][i]])
    else:
        Test_precision += sum([True for i in range(len(batch_text)) if int(test_prec[i][0]) in test_data[2][nu*len(batch_text):(nu+1)*len(batch_text)][i]])

    for ij in range(len(batch_text)):
        for_do.append([batch_y[ij],test_prec[ij]])

    nu+=1

print("Test_Precision : %.05f" % (Test_precision/len(test_data[0])))

# in this case : test accuracy is 0.45298

