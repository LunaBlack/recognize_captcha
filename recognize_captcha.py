#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import os
import colorsys
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy import ndimage, misc
from sklearn import neighbors, svm


#验证码图片在pic文件夹中，共50张
#这些图片的特点：1、多颜色，且验证码不一定为最深的颜色；2、无噪点，有干扰线



# 获取图片的主要颜色(假定其为背景色)
def get_dominant_color(pic):
    #颜色模式转换，以便输出rgb颜色值
    pic = pic.convert('RGBA')   
    #生成缩略图，减少计算量，减小cpu压力
    pic.thumbnail((200, 200))
    
    max_score = None
    dominant_color = None
     
    for count, (r, g, b, a) in pic.getcolors(pic.size[0] * pic.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue
         
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]       
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)        
        y = (y - 16.0) / (235 - 16)
         
        # 忽略高亮色
        if y > 0.9:
            continue
         
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count
         
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
     
    return dominant_color



#清除单个孤立点
def Clear_Point(im):
    for j in range(1,(im.shape[1]-1)):
        for i in range(1,(im.shape[0]-1)):
            if im[i][j]==1 and im[i][j-1]==0\
               and im[i][j+1]==0 and im[i-1][j-1]==0\
               and im[i-1][j]==0 and im[i-1][j+1]==0\
               and im[i+1][j-1]==0 and im[i+1][j]==0\
               and im[i+1][j+1]==0:
                im[i][j] = 0
    return im



# 将图片二值化
def binaryzation(im):

    #改变背景色
    r, g, b = get_dominant_color(im)
    r0, g0, b0 = (0, 0, 0) #黑色
    for i in xrange(im.size[0]):
        for j in xrange(im.size[1]):
            r1, g1, b1 = im.getpixel((i,j))
            if r1==r and g1==g and b1==b:
                im.putpixel((i,j), (r0,g0,b0))

    #加强对比，并转为灰度图
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(10)
    im = im.convert('L')

    #用scipy库进行开运算，去除干扰线，保存
    imarray = np.asarray(im, dtype=np.uint8) #白色为1,即字符
    imarray = ndimage.binary_erosion(imarray)

    imarray = Clear_Point(imarray) #清除周围8个像素都是黑(白)色的孤立噪点
    #imarray = ndimage.binary_opening(imarray, structure=np.ones((1,1)))

    #imarray = np.invert(imarray)
    #misc.imsave('.//im//%s' % pic, imarray)
    return imarray



# 根据去噪后的黑白图，针对白色部分，获取原图的颜色排序
def get_color(im, new_im):
    #颜色模式转换，以便输出rgb颜色值
    im = im.convert('RGB')   
    #生成缩略图，减少计算量，减小cpu压力
    im.thumbnail((200, 200))
    
    color = {}
    for count, (r, g, b) in im.getcolors(im.size[0] * im.size[1]):
        color[(r,g,b)] = 0

    for i in range(im.size[0]):
        for j in range(im.size[1]):  
            if new_im.getpixel((i,j)) != 0:
                r0, g0, b0 = im.getpixel((i,j))
                color[(r0,g0,b0)] += 1

    return color



# 改变图片的颜色，利用颜色不同
def change_image(im, new_im, pic):
    rgb0 = (0, 0, 0) #黑色
    rgb1 = (255, 255, 255) #白色
    
    color = get_color(im, new_im)
    score = sorted(color.iteritems(),key=lambda t:t[1],reverse=True)

    #改变字符颜色（为方便处理，黑底白字）
    for times in range(4): #防止干扰线分数过高
        (r,g,b) = score.pop(0)[0]
        for i in xrange(im.size[0]):
            for j in xrange(im.size[1]):
                r2, g2, b2 = im.getpixel((i,j))
                if r2==r and g2==g and b2==b:
                    im.putpixel((i,j), rgb1)

    #改变干扰线颜色
    while(score):
        (r,g,b) = score.pop(0)[0]
        for i in xrange(im.size[0]):
            for j in xrange(im.size[1]):
                r2, g2, b2 = im.getpixel((i,j))
                if r2==r and g2==g and b2==b:
                    im.putpixel((i,j), rgb0)

    im.save('.//im//%s' % pic)
    im = im.convert('1')
    return im



#依据图片像素颜色计算X轴投影
def caculate_x(im):
    Image_Value=[]
    for i in range(im.size[0]):
        Y_pixel=0
        for j in range(im.size[1]):
            if im.getpixel((i,j)) == 255: #白色，即字符
                temp_value = 1
            else:
                temp_value = 0
            Y_pixel = Y_pixel + temp_value
        Image_Value.append(Y_pixel)
    return Image_Value



#依据图片像素颜色计算Y轴投影
def caculate_y(im):
    Image_Value=[]
    for j in range(im.size[1]):
        X_pixel=0
        for i in range(im.size[0]):
            if im.getpixel((i,j)) == 255: #白色，即字符
                temp_value = 1
            else:
                temp_value = 0
            X_pixel = X_pixel + temp_value
        Image_Value.append(X_pixel)
    return Image_Value



# 切割图片
def cut(im, pic):
    for k in range(4):
        im_cut = im.crop((k*30+5,0,k*30+35,50))
        ylist = caculate_y(im_cut)
        y_start = y_end = 0

        for n, i in enumerate(ylist):
            if i != 0:
                y_start = n
                break

        ylist.reverse()
        for n, i in enumerate(ylist):
            if i != 0:
                y_end = 50 - n
                break

        num1 = (50 - (y_end - y_start)) / 2
        num2 = 50 - (y_end - y_start) - num1
        im_cut = im_cut.crop((0,y_start-num1,30,y_end+num2))
        im_cut.save('.//cut_im//%s_%d.png' % (pic[0:4],k))



# 针对50个验证码，提取出所有待识别的单个字符
def extract_str():
    for pic in os.listdir('.//pic/'):
        im = Image.open('.//pic//%s' % pic) #原图
        
        imarray = binaryzation(im) #初次处理后的图，即去噪后的图
        new_im = Image.fromarray(np.asarray(imarray, dtype=np.uint8))
        final_im = change_image(im, new_im, pic) #提取字符后的图
        
        cut(final_im, pic)



# 获取model里面的字符特征矩阵和标签矩阵
def get_data():
    data = []
    labels = []
    
    for pic in os.listdir('.//model/'):
        im = np.array(Image.open('.//model//%s' % pic).convert('L'))
        im[im==255] = 1
        
        temp = []
        for i in im:
            temp += list(i)
        data.append(temp)
        labels.append(pic[-5])
        
    return data, labels



# 用KNN算法进行预测
def knn_predict(data, labels, k):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(data, labels)

    error = 0
    total = 0
    error_dict = {}
    
    for pic in os.listdir('.//cut_im/'):
        im = np.array(Image.open('.//cut_im//%s' % pic).convert('L'))
        im[im==255] = 1
        
        newdata = []
        for i in im:
            newdata += list(i)
            
        predict = knn.predict(np.array(newdata))[0]
        fact = pic[int(pic[-5])]

        total += 1
        error_dict.setdefault(fact, [0,0])
        error_dict[fact][0] += 1
        if predict != fact:
            error += 1
            error_dict[fact][1] += 1

    print "k取%d时, knn错误率为: " % k, error*1.0/total
##    for key in error_dict.keys():
##        print "字符%s的错误率为: " % key, error_dict[key][1]*1.0/error_dict[key][0]



# 用svm_rbf算法进行预测
def svm_rbf_predict(data, labels, k):
    svm_rbf = svm.SVC(kernel='rbf', gamma=k)
    svm_rbf.fit(data, labels)

    error = 0
    total = 0
    error_dict = {}
    
    for pic in os.listdir('.//cut_im/'):
        im = np.array(Image.open('.//cut_im//%s' % pic).convert('L'))
        im[im==255] = 1
        
        newdata = []
        for i in im:
            newdata += list(i)
            
        predict = svm_rbf.predict(np.array(newdata))[0]
        fact = pic[int(pic[-5])]

        total += 1
        error_dict.setdefault(fact, [0,0])
        error_dict[fact][0] += 1
        if predict != fact:
            error += 1
            error_dict[fact][1] += 1

    print "k取%s时, svm_rbf错误率为: " % str(k), error*1.0/total
    


# 用svm_linear算法进行预测
def svm_linear_predict(data, labels):
    svm_linear = svm.SVC(kernel='linear')
    svm_linear.fit(data, labels)

    error = 0
    total = 0
    error_dict = {}
    
    for pic in os.listdir('.//cut_im/'):
        im = np.array(Image.open('.//cut_im//%s' % pic).convert('L'))
        im[im==255] = 1
        
        newdata = []
        for i in im:
            newdata += list(i)
            
        predict = svm_linear.predict(np.array(newdata))[0]
        fact = pic[int(pic[-5])]

        total += 1
        error_dict.setdefault(fact, [0,0])
        error_dict[fact][0] += 1
        if predict != fact:
            error += 1
            error_dict[fact][1] += 1

    print "svm_linear错误率为: ", error*1.0/total




if __name__ == '__main__':
    try:
        os.mkdir('im')
        os.mkdir('cut_im')
    except:
        pass
    
    extract_str()
    
    data, labels = get_data()
    
    for k in [1,3,5,10]:
        knn_predict(np.array(data), np.array(labels), k)

    for k in [0, 0.001, 0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.5, 1]:
        svm_rbf_predict(np.array(data), np.array(labels), k)

    svm_linear_predict(np.array(data), np.array(labels))
            
