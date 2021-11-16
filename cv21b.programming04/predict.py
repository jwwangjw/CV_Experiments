import numpy as np
from PIL import Image
import os
from facenet import Facenet

if __name__ == "__main__":
    fp=open('./result_test.txt','w')
    model = Facenet()
        
    #while True:
    '''image_1 = input('Input image_1 filename:')

        image_1 = Image.open(image_1)'''
        #except:
         #   print('Image_1 Open Error! Try again!')
          #  continue
    galleryList=[]
    for item in os.listdir('D:/学习数据/计算机视觉/作业四/gallery'):
        print(item)
        list_t=item.split('.')
        galleryList.append(int(list_t[0]))
    galleryList.sort()
    print(galleryList)
    imageList=[]
    for item in os.listdir('D:/学习数据/计算机视觉/作业四/test'):
        print(item)
        imageList.append(item)
    imageList.sort()
    print(imageList)
    str1='D:/学习数据/计算机视觉/作业四/test'
    str2='D:/学习数据/计算机视觉/作业四/gallery'
    for i in range(len(imageList)):
        image1 = str1 + '/' + imageList[i]
        image1=Image.open(image1)
        list_nt=[]
        for j in range(len(galleryList)):
            image2=str2+'/'+str(galleryList[j])+'.jpg'
            image2=Image.open(image2)
            probability = model.detect_image(image1, image2)
            list_nt.append(probability)
        min_res=list_nt.index(min(list_nt))
        print(str(i)+' '+str(min_res))
        fp.write(imageList[i]+' '+str(min_res)+'\n')


    '''image_2 = 
            try:
                image_2 = Image.open(image_2)
            except:
                print('Image_2 Open Error! Try again!')
                continue
        
            probability = model.detect_image(image_1,image_2)
            print(probability)'''
