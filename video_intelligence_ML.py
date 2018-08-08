from PIL import Image
import cv2
import os
import numpy as np
import math
import sys
from classify_image import *
import tensorflow as tf
import argparse
import scipy.cluster.hierarchy as hcluster


video = 'julian_callari_trumpwall.mp4'
frames = []
def createFrames(video):
    print("capturing video frames")
    cap = cv2.VideoCapture(video)
    frameRate = math.floor(cap.get(5)) #frame rate
    id = 0
    #nm = FLAGS.video_file.split('.')[0]
    #os.mkdir('D:/videos_frames/' + nm+ '/')
    #save_path = 'D:/videos_frames/' + nm + '/'
    while(True):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId*2 % frameRate == 0):
            frm = Image.fromarray(np.uint8(frame[:,:,::-1])).convert('RGB')
            
            #frm.save(save_path + 'frame' + str(id) + '.png')
            #id = id+1
            frames.append(frm)
    print("end of frames")   
    cap.release()
    cv2.destroyAllWindows()
    print(len(frames))
createFrames(video)
print(len(frames))

frames_embedds = []
def create_embeddings(frames):
    print("create Embeddings", type(frames), len(frames))
    maybe_download_and_extract()
    sess = createSession()
    
    for img in frames:
        embd = run_inference_on_image(img, sess)
        e = embd.reshape(2048,)
        print("shape", e.shape)
        #os.system('C:/Users/user/AppData/Local/Programs/Python/Python36/python.exe', classify_image)
        #embd = os.system('%s %s %s' % ('C:/Users/user/AppData/Local/Programs/Python/Python36/python.exe', classify_image, '--image-file'+img)               )
        #embd = classify(img)
        #print("embedding: ", embd, type(embd), type(embd[0]))
        frames_embedds.append(e)

    x=0
    print("number of embeddings: "  + str(len(frames_embedds)))
    #return(create_cluster_embeddings(frames_embedds))
create_embeddings(frames)


thresh = 10
clusters = []
def create_cluster_embeddings(frames_embeddings):
    global clusters
    print('frames_embeddings', type(frames_embeddings))
    # Run forward pass to calculate embeddings
    X = np.asarray(frames_embeddings)
    print("x",X, type(X), X.size)
    #for i, img in enumerate(frames_embeddings):
     #   print("embedds", i, img, type(i), type(img))
    clusters = hcluster.fclusterdata(X, t=thresh, criterion="distance")
    print("CLUSTERS : ", clusters)
create_cluster_embeddings(frames_embedds)

print("clusters:", clusters)
dict = {}
for i, key in enumerate(clusters):
    if key in dict:
        dict[key].append(i)
    else:
        dict[key] = [i]
print(dict)

import matplotlib.pyplot as plt
for i,j in dict.items():
    print(i,j)
    print("CLUSTER" + str(i))
    for val in j:
        print(val)
        imgplot = plt.imshow(frames[val])
        plt.show()
        