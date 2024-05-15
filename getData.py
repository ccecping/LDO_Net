import os
import cv2
import numpy as np
import h5py
#from keras.utils import np_utils, conv_utils
##from keras import backend as K
##
##from keras.models import Model
##from keras.layers import Flatten, Dense, Input
##from keras.optimizers import Adam
##from keras.applications.resnet50 import ResNet50
##from keras import backend as K
##from keras.preprocessing.image import load_img,img_to_array
from keras.utils import to_categorical
from PIL import Image

#from keras.utils import conv_utils
Width=224
Height=224
def get_name_list(filepath):  # 获取各个类别的名字
    pathDir = os.listdir(filepath)
    out = []
    for allDir in pathDir:
        if os.path.isdir(os.path.join(filepath, allDir)):
            # child = allDir.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
            print(allDir)
            out.append(allDir)
    return out

def eachFile(filepath):  # 将目录内的文件名放入列表中
    pathDir = os.listdir(filepath)
    out = []
    for allDir in pathDir:
        # child = allDir.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
        out.append(allDir)
    #print(out)
    return out

def get_predict_data(img_dir,h5_dir,data_name, train_left=0.0, train_right=0.8, train_all=0.8, resize=True, data_format=None,t='',num_classes=3,channels=3):  # 从文件夹中获取图像数据
    pic_dir_set = eachFile(img_dir)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0
    labels = []
    for pic_dir in pic_dir_set:
        #print('label is',label)
        y = to_categorical(label,num_classes=num_classes)
        
        #print('one-hot is ',y)
        #print('*****', pic_dir)
        labels.append(pic_dir) 
    file_name = os.path.join(h5_dir, data_name + t + ".h5")
    print(file_name)
    if os.path.exists(file_name):  # 判断之前是否有存到文件中
        f = h5py.File(file_name, 'r')
        if t == 'predict':
            X_train = f['X_train'][:]
            y_train = f['y_train'][:]
            f.close()
            print('X_train.shape is ++++',len(X_train))
            print('y_train.shape is ++++',len(y_train))
            return (X_train, y_train,labels)
        elif t == 'test':
            X_test = f['X_test'][:]
            y_test = f['y_test'][:]
            f.close()
            return (X_test, y_test,labels)
        else:
            return
    #data_format = normalize_data_format(data_format)
    
    for pic_dir in pic_dir_set:
        #print('label is',label)
        y = to_categorical(label,num_classes=num_classes)
        #print('one-hot is ',y)
        #print('*****', pic_dir)
        #labels.append(pic_dir)
        #print('****', os.path.join(img_dir, pic_dir))
        if not os.path.isdir(os.path.join(img_dir, pic_dir)):
            continue
        pic_set = eachFile(os.path.join(img_dir, pic_dir))
        print('训练集是',pic_set,' length is',len(pic_set)) 
        pic_index = 0
        train_count = int(len(pic_set) * train_all)
        train_l = int(len(pic_set) * train_left)
        train_r = int(len(pic_set) * train_right)
        
        for pic_name in pic_set:
            if not os.path.isfile(os.path.join(img_dir, pic_dir, pic_name)):
                continue
            #img = cv2.imread(os.path.join(img_dir, pic_dir, pic_name))
            img = cv2.imdecode(np.fromfile(os.path.join(img_dir, pic_dir, pic_name), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if img is None:
                continue
            if (resize):
                
                img = cv2.resize(img, (Width, Height))
                #img = img.convert("L")
                #print(img.shape)
                if len(img.shape)>2 and img.shape[2]==4:
                    img = Image.fromarray(np.uint8(img))
                    img = img.convert("RGB")
                    img = np.array(img)
                img = img.reshape(-1, Width, Height, channels)
            if (pic_index < train_count):
                if t == 'predict':
                    if (pic_index >= train_l and pic_index < train_r):
                        X_train.append(img)
                        y_train.append(y)
            else:
                if t == 'test':
                    X_test.append(img)
                    y_test.append(y)
            pic_index += 1
        if len(pic_set) != 0:
            label += 1
    print('X_train.shape is ?',len(X_train))
    f = h5py.File(file_name, 'w')
    if t == 'predict':
        print('X_train.shape before concatenate is  ',len(X_train),X_train[0].shape)
        X_train = np.concatenate(X_train, axis=0)
        X_train = np.array(X_train)
        print('X_train.shape after concatenate is  ',len(X_train),X_train.shape)
        y_train = np.array(y_train)
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.close()
        print('X_train.shape is ',len(X_train))
        print('y_train.shape is ',len(y_train))
        return (X_train, y_train,labels)
    elif t == 'test':
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.array(y_test)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.close()
        return (X_test, y_test,labels)
    else:
        return
def get_predict_data_transformer(img_dir,h5_dir,data_name, Width, Height,train_left=0.0, train_right=0.8, train_all=0.8, resize=True, data_format=None,t=''):  # 从文件夹中获取图像数据
    file_name = os.path.join(h5_dir, data_name + t + ".h5")
    print(file_name)
    if os.path.exists(file_name):  # 判断之前是否有存到文件中
        f = h5py.File(file_name, 'r')
        if t == 'predict':
            X_train = f['X_train'][:]
            y_train = f['y_train'][:]
            f.close()
            return (X_train, y_train)
        elif t == 'test':
            X_test = f['X_test'][:]
            y_test = f['y_test'][:]
            f.close()
            return (X_test, y_test)
        else:
            return
    #data_format = normalize_data_format(data_format)
    pic_dir_set = eachFile(img_dir)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0
    for pic_dir in pic_dir_set:
        print('label is',label)
        #y = to_categorical(label,num_classes=3)
        #print('one-hot is ',y)
        print('*****', pic_dir)
        print('****', os.path.join(img_dir, pic_dir))
        if not os.path.isdir(os.path.join(img_dir, pic_dir)):
            continue
        pic_set = eachFile(os.path.join(img_dir, pic_dir))

        pic_index = 0
        train_count = int(len(pic_set) * train_all)
        train_l = int(len(pic_set) * train_left)
        train_r = int(len(pic_set) * train_right)
        for pic_name in pic_set:
            if not os.path.isfile(os.path.join(img_dir, pic_dir, pic_name)):
                continue
            #img = cv2.imread(os.path.join(img_dir, pic_dir, pic_name))
            img = cv2.imdecode(np.fromfile(os.path.join(img_dir, pic_dir, pic_name), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            
            if img is None:
                continue
            if (resize):
                
                img = cv2.resize(img, (Width, Height))
                #print(img.shape)
                if img.shape[2]==4:
                    img = Image.fromarray(np.uint8(img))
                    img = img.convert("RGB")
                    img = np.array(img)
                img = img.reshape(-1, Width, Height, 3)
            if (pic_index < train_count):
                if t == 'predict':
                    if (pic_index >= train_l and pic_index < train_r):
                        X_train.append(img)
                        y_train.append(label)
            else:
                if t == 'test':
                    X_test.append(img)
                    y_test.append(label)
            pic_index += 1
        if len(pic_set) != 0:
            label += 1
    print('X_train.shape is ',len(X_train))
    f = h5py.File(file_name, 'w')
    if t == 'predict':
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.array(y_train)
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.close()
        return (X_train, y_train)
    elif t == 'test':
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.array(y_test)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.close()
        return (X_test, y_test)
    else:
        return

def get_predict_data_mobileNet(img_dir,h5_dir,data_name, train_left=0.0, train_right=0.8, train_all=0.8, resize=True, data_format=None,
             t=''):  # 从文件夹中获取图像数据
    pic_dir_set = eachFile(img_dir)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0
    labels = []
    for pic_dir in pic_dir_set:
        #print('label is',label)
        y = to_categorical(label,num_classes=3)
        
        #print('one-hot is ',y)
        #print('*****', pic_dir)
        labels.append(pic_dir) 
    file_name = os.path.join(h5_dir, data_name + t + ".h5")
    print(file_name)
    if os.path.exists(file_name):  # 判断之前是否有存到文件中
        f = h5py.File(file_name, 'r')
        if t == 'predict':
            X_train = f['X_train'][:]
            y_train = f['y_train'][:]
            f.close()
            return (X_train, y_train,labels)
        elif t == 'test':
            X_test = f['X_test'][:]
            y_test = f['y_test'][:]
            f.close()
            return (X_test, y_test,labels)
        else:
            return
    #data_format = normalize_data_format(data_format)
    
    for pic_dir in pic_dir_set:
        #print('label is',label)
        y = to_categorical(label,num_classes=3)
        #print('one-hot is ',y)
        #print('*****', pic_dir)
        #labels.append(pic_dir)
        #print('****', os.path.join(img_dir, pic_dir))
        if not os.path.isdir(os.path.join(img_dir, pic_dir)):
            continue
        pic_set = eachFile(os.path.join(img_dir, pic_dir))

        pic_index = 0
        train_count = int(len(pic_set) * train_all)
        train_l = int(len(pic_set) * train_left)
        train_r = int(len(pic_set) * train_right)
        for pic_name in pic_set:
            if not os.path.isfile(os.path.join(img_dir, pic_dir, pic_name)):
                continue
            #img = cv2.imread(os.path.join(img_dir, pic_dir, pic_name))
            img = cv2.imdecode(np.fromfile(os.path.join(img_dir, pic_dir, pic_name), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            
            if img is None:
                continue
            if (resize):
                
                img = cv2.resize(img, (Width, Height))
                #print(img.shape)
                if img.shape[2]==4:
                    img = Image.fromarray(np.uint8(img))
                    img = img.convert("RGB")
                    img = np.array(img)
                img = img.reshape(-1, Width, Height, 3)
            if (pic_index < train_count):
                if t == 'predict':
                    if (pic_index >= train_l and pic_index < train_r):
                        X_train.append(img)
                        y_train.append(y)
            else:
                if t == 'test':
                    X_test.append(img)
                    y_test.append(y)
            pic_index += 1
        if len(pic_set) != 0:
            label += 1
    print('X_train.shape is ',len(X_train))
    f = h5py.File(file_name, 'w')
    if t == 'predict':
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.array(y_train)
        y_train = reshape(y_train)
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.close()
        return (X_train, y_train,labels)
    elif t == 'test':
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.array(y_test)
        y_test = reshape(y_test)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.close()
        return (X_test, y_test,labels)
    else:
        return

def reshape(ys):
    new_shape = (ys.shape[0], 1, 1, ys.shape[-1])
    return ys.reshape(new_shape)
##get_name_list('../CT/test/')
##eachFile('../CT/test/')

#(X_train,y_train) = get_predict_data('../CT/train/','CT',t='predict')
##(X_train,y_train) = get_predict_data('croppedImage/Both/','CT',t='predict')
##print(X_train.shape)
main_path = 'dataset/combined2types/'
data_path = 'datafiles/combined2types/'
(x_train,y_train,labels) = get_predict_data(main_path,data_path,'c2_train',t='predict',num_classes=2,channels=3)
(x_test,y_test,_) = get_predict_data(main_path,data_path,'c2_test',t='test',num_classes=2,channels=3)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
