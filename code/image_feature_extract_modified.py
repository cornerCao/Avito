import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
import zipfile

from collections import Counter
from pprint import pprint

from PIL import Image
import cv2
from skimage import feature
import operator

import matplotlib.pyplot as plt
#%matplotlib inline
print(os.listdir("../input"))

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Create symbolic links for trained models.
# Thanks to Lem Lordje Ko for the idea
# https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied
models_symlink = os.path.join(cache_dir, 'models')
if not os.path.exists(models_symlink):
    os.symlink('/kaggle/input/keras-pretrained-models/', models_symlink)


def image_classify(model, pak, img, top_n=3):
    """Classify image and return top matches."""
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    return pak.decode_predictions(preds, top=top_n)[0]


def classify_and_plot(image_path, classify=False):
    """Classify an image with different models.
    Plot it and its predicitons.
    """
    img = Image.open(image_path)
    cv_img = cv2.imread(image_path)

    if classify:
        resnet_preds = image_classify(resnet_model, resnet50, img)
        xception_preds = image_classify(xception_model, xception, img)
        inception_preds = image_classify(inception_model, inception_v3, img)
        preds_arr = [('Resnet50', resnet_preds), ('xception', xception_preds), ('Inception', inception_preds)]
    else:
        preds_arr = None
    return (img, cv_img, preds_arr, image_path)


def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = {}
    for pixel in img.getdata():
        if pixel not in palatte:
            palatte[pixel] = 0
        palatte[pixel] += 1
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent


def perform_color_analysis(img, flag):
    im = img#.convert("RGB")

    # cut the images into two halves as complete average may give bias results
    size = im.size
    # halves = (size[0]/2, size[1]/2)
    # im1 = im.crop((0, 0, size[0], halves[1]))
    # im2 = im.crop((0, halves[1], size[0], size[1]))

    #  try:
    light_percent1, dark_percent1 = color_analysis(im)
    #    light_percent2, dark_percent2 = color_analysis(im2)
    #  except Exception as e:
    #      return None
   # light_percent = (light_percent1 + light_percent2)/2
   # dark_percent = (dark_percent1 + dark_percent2)/2

    if flag == 'black':
        return dark_percent1
    elif flag == 'white':
        return light_percent1
    else:
        return None


def average_pixel_width(img):
    im = img
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100


def get_blurrness_score(image):
    #path =  img_path
    #image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


def get_data_from_image(dat, classify=False):
    # plt.imshow(dat[0])
    img = dat[0]
    img_path = dat[3]
    cv_image = dat[1]
    img_size = [dat[0].size[0], dat[0].size[1]]
    (means, stds) = cv2.meanStdDev(dat[1])
    mean_color = np.mean(dat[1].flatten())
    std_color = np.std(dat[1].flatten())
    color_stats = np.concatenate([means, stds]).flatten()
    if classify:
        scores = [i[1][0][2] for i in dat[2]]
        labels = [i[1][0][1] for i in dat[2]]
    else:
        scores = [0, 0, 0]
        labels = [0, 0, 0]
    dullness = [perform_color_analysis(img, 'black')]
    whiteness = [perform_color_analysis(img, 'white')]
    avg_pixel_width = [average_pixel_width(img)]
    blurrness = [get_blurrness_score(cv_image)]
    #df = pd.DataFrame([img_size + [mean_color] + [std_color] + color_stats.tolist() + scores + labels],
    #                    columns = ['img_size_x', 'img_size_y', 'img_mean_color', 'img_std_color', 'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'image_green_std', 'image_red_std', 'Resnet50_score', 'xception_score', 'Inception_score', 'Resnet50_label', 'xception_label', 'Inception_label'])
    df_row = img_size + [mean_color] + [std_color] + color_stats.tolist() + scores + labels + dullness + whiteness + avg_pixel_width + blurrness
    return df_row

print("start fetch feature")
NUM_IMAGES_TO_EXTRACT = 10
classify = False

#images_dir = os.path.expanduser(os.path.join('~', 'avito_images'))
#if not os.path.exists(images_dir):
 #   os.makedirs(images_dir)
if classify:
    resnet_model = resnet50.ResNet50(weights='imagenet')
    inception_model = inception_v3.InceptionV3(weights='imagenet')
    xception_model = xception.Xception(weights='imagenet')
    print("finish import model")
else:
    print("finish import packages...\n Start extracting features...\n")
image_feature_df = pd.DataFrame(columns = ['img_name','img_size_x', 'img_size_y', 'img_mean_color', 'img_std_color', 'img_blue_mean', 'img_green_mean', 'img_red_mean',
                                            'img_blue_std', 'image_green_std', 'image_red_std', 'Resnet50_score', 'xception_score', 'Inception_score', 'Resnet50_label', 'xception_label', 
                                            'Inception_label','dullness','whiteness','avg_pixel_width','blurrness'])

fileno = 1

bad_images = ['b98b291bd04c3d92165ca515e00468fd9756af9a8f1df42505deed1dcfb5d7ae.jpg',
              '8513a91e55670c709069b5f85e12a59095b802877715903abef16b7a6f306e58.jpg',
              '60d310a42e87cdf799afcd89dc1b11ae3fdc3d0233747ec7ef78d82c87002e83.jpg',
              '4f029e2a00e892aa2cac27d98b52ef8b13d91471f613c8d3c38e3f29d4da0b0c.jpg']

with zipfile.ZipFile('../input/train_jpg_%s.zip' % fileno, 'r') as train_zip:
    files_in_zip = sorted(train_zip.namelist())
    #每次取1个
    for idx, file in enumerate(files_in_zip):
        if file.endswith('.jpg'):
            try:
                images_dir = '../working/avito_images%s/' % fileno
                train_zip.extract(file, path=images_dir)
                image_files = [x.path for x in os.scandir(images_dir)]
                for image_file in image_files:
                    if image_file in bad_images:
                        continue  # there are 4 bad images, so continue when meet them.
                    dat = classify_and_plot(image_file, classify=False)
                    df_row = [image_file.replace(images_dir,'')] + get_data_from_image(dat)
                    image_feature_df.loc[len(image_feature_df)] = df_row
                    #  print(df_row)
                    os.remove(image_file)
                #  print(image_file)
                if len(image_feature_df) % 200 == 0:
                    print(len(image_feature_df))
            except(IOError, StopIteration, RuntimeError):
                print("WARNING: Error occur in line 188, maybe no image found...")

image_feature_df.to_csv('../input/image_features_%s.csv'%fileno, index=False)

