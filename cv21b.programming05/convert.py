import json
import random
import torch

with open('./train.json','r',encoding='utf-8')as fp:
    json_data = json.load(fp)
d_order=dict(sorted(json_data.items(),key=lambda x:int(x[0].split('.')[0]),reverse=False))  # 按字典集合中，每一个元组的第二个元素排列。
labels=[]
names=[]                                                           # x相当于字典集合中遍历出来的一个元组。
for key in d_order:
    tmp=d_order[key]
    labels.append(tmp)
    for j in range(len(tmp)):
        names.append(tmp[j])
num=[]
for i in range(len(names)):
    num.append(i)
dict_lable=dict(zip(names,num))
names_o=[]
for key in dict_lable:
    names_o.append(key)
print(labels)
list_t=[]
for i in range(len(labels)):
    list_t.append(str(i)+'.jpg')
dict_lables=dict(zip(list_t,labels))
print(dict_lables)
names_o.append(' ')
with open('./data/char_std_5990.txt','w') as f:
    for i in range(len(names_o)):
        f.write(names_o[i]+'\n')
rate=0.2
random.shuffle(list_t)
with open('./train.txt','w') as fp:
    for i in range(int(len(list_t)*(1-rate))):
        fp.write('D:/datasets/train/'+list_t[i]+' '+dict_lables[list_t[i]]+'\n')
with open('./test.txt','w') as l:
    for i in range(int(len(list_t)*(1-rate)),len(list_t)):
        l.write('D:/datasets/train/'+list_t[i]+' '+dict_lables[list_t[i]]+'\n')

print(torch.cuda.get_device_name(0))
#cd crnn_seq2seq_ocr.Pytorch