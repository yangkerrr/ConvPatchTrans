import os
import cv2
import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import random
from network.shrnet import SHR
from dataset.CVSIdata import loaders_default
import argparse

def accuracy(dataloaders, net, num_classes,  disp=False):

    net.eval()
    correct = 0
    total = 0
    class_correct = list(0 for _ in range(num_classes))
    class_total = list(0 for _ in range(num_classes))
    conf_matrix = np.zeros([num_classes, num_classes])  # (i, j): i-Gt; j-Pr
    with paddle.no_grad():
        for loader in iter(dataloaders):
            for data in iter(loader):
                imgs, labels = data
                outputs = net(imgs)
                predicted = paddle.argmax(outputs, axis = 1)
                total += labels.shape[0]
                correct += (predicted == labels).astype("int").sum()
                c = (predicted == labels).astype("int").squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    try:
                        class_correct[label] += c[i]
                    except:
                        class_correct[label] += c
                    class_total[label] += 1
                    conf_matrix[label, predicted[i]] += 1
        accr = correct / total
    if disp:
        print('Total number of images={}'.format(total))
        print('Total number of correct images={}'.format(correct))

    return accr, class_correct, class_total, conf_matrix

def calculate_multiloss(net,data, i, f_loss, weight):
    images, labels= data
    ys = net(images)
    loss_ce = 0.0
    f_loss0 = nn.NLLLoss()
    for j, y in enumerate(ys):
        y=y.astype("float")
        try:
            loss_ce = loss_ce + f_loss(y,labels)* weight[j]
        except:
            print("y",y)
            print("label",labels)        
    return loss_ce



paddle.set_device("gpu:0")
parser = argparse.ArgumentParser()
parser.add_argument('--choice', type=str, default="CVSI",help='choos dataset')
parser.add_argument('--weight', type=int, default="0",help='choos dataset')
opt = parser.parse_args()
bs = 32

if opt.choice == "CVSI":
    classes = ('Arabic', 'Bengali', 'English', 'Gujrathi', 'Hindi', 'Kannada', 'Oriya', 'Punjabi', 'Tamil', 'Telegu')
    loaders_train = loaders_default(istrainset=True,  batchsize=bs)
    loaders_val = loaders_default(istrainset=False, batchsize=bs)
elif opt.choice == "SIW":
    classes = ('Arabic', 'Cambodian', 'Chinese', 'English', 'Greek', 'Hebrew', 'Japanese', 'Kannada',
            'Korean', 'Mongolian', 'Russian', 'Thai', 'Tibetan')
elif opt.choice == "MLT":
    classes = ("Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols")
    loaders_train = loaders_default(istrainset=True, classes=classes, batchsize=bs)
    loaders_val = loaders_default(istrainset=False, classes=classes, batchsize=bs)

weight_num = opt.weight
num_classes = len(classes)
'''
net = SHR(num_classes)

optimizer = paddle.optimizer.Momentum(learning_rate=1e-2, momentum=0.9,parameters=net.parameters())
num_epoch=500

lr_list = [1e-1]
lr_min=1e-5
i_list = []
avg_loss_list = []
accr_list = []

accr_best = 0.0
dc = 6
model_param = "params/cvsiequal.pdparams"
f_loss = paddle.nn.CrossEntropyLoss(reduction='mean')
'''
print("***Start training!***")

weights = [[0.1,0.1,0.1,1]]
net = SHR(num_classes)
#optimizer = paddle.optimizer.Momentum(learning_rate=1e-3, momentum=0.9,parameters=net.parameters())
optimizer = paddle.optimizer.Adam(learning_rate=1e-4,parameters=net.parameters())
num_epoch=200

lr_list = [1e-4]
lr_min=1e-5
i_list = []
avg_loss_list = []
accr_list = []

accr_best = 0.0
dc = 6
model_param = "params/cvsibest.pdparams"
f_loss = paddle.nn.CrossEntropyLoss(reduction='mean')
print("--------------------------------------------")
print(weight_num,)
for epoch in range(num_epoch):
    net.train()
    running_loss = 0.0
    random.shuffle(loaders_train)
    i0 = 0
    for loader in iter(loaders_train):
        for i2, data in enumerate(loader):
            i = i0 + i2
            loss = calculate_multiloss(net, data, i, f_loss, weight=weights[weight_num])
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            running_loss += loss.item()       
        i0 = i + 1
    avg_loss = running_loss / i
    test_net = net
    
    accr_cur, _, _, _ = accuracy(loaders_val, test_net, num_classes)
    accr_list.append(accr_cur)
    if accr_best < accr_cur:
        net_best = net
        accr_best = accr_cur
        loss_best = avg_loss
        # Save the trained Net
        print("saving...", end=', ')
        paddle.save(net_best.state_dict(), model_param)
        #f.write(str(accr_best)+'\n')
    print('[%2d,%4d] lr=%.5f loss: %.4f accr(val): %.3f%%' %
        (epoch + 1, i + 1, lr_list[0], avg_loss, accr_cur * 100))

    #print('[%2d,%4d] lr=%.5f loss: %.4f accr(val): %.3f%%' %
    #        (epoch + 1, i + 1, lr_list[0], avg_loss, accr_cur * 100))
    i_list.append(epoch)
    avg_loss_list.append(avg_loss)
    if len(avg_loss_list) > 1 and ((1 - avg_loss_list[-1] / (avg_loss_list[-2] + 1e-10) < 0.005) or
                                    (avg_loss < 0.005 and avg_loss_list[-2] - avg_loss_list[-1] < 0.0003)):
        dc -= 1
        if avg_loss > 0.1 and avg_loss_list[-1] / (avg_loss_list[-2] + 1e-10) - 1 > 0.6:
            dc += 1
        if dc == 0:
            lr_list = [0.3 * i for i in lr_list]
            net.set_state_dict(paddle.load(model_param))
            dc = 16 if lr_list[0] > 1e-3 else 10
            if max(lr_list) < lr_min:
                lr_list = [1e-2 for i in lr_list]
            optimizer = paddle.optimizer.Momentum(learning_rate=1e-3, momentum=0.9,parameters=net.parameters())
print("----------------------------------------------")

    #f.write("---------------------------------------\n")
    #f.close()
# plot loss-iter figure
#print("best_loss=%.3f, best_accr=%.3f %%, model params %s saved " % (loss_best, accr_best * 100, model_param))
