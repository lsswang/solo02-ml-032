"""Ordinary disjoint-mask decoder; later slot wins overlap."""
import numpy as np
from scipy.special import softmax,expit

def decode(class_logits,mask_logits,depth,class_threshold=0.5,mask_threshold=0.5):
    p=softmax(np.asarray(class_logits,float),axis=1);v=expit(np.asarray(mask_logits,float));owner=np.full(v.shape[1:],-1,int)
    active=[]
    for i in range(len(v)):
        if 1-p[i,0]>=class_threshold:
            owner[v[i]>=mask_threshold]=i;active.append(i)
    return {'owner':owner.tolist(),'active':active}
