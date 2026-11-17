"""Ordinary fully labeled independent instance matching loss."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

def loss(class_logits,mask_logits,depth,targets,beta=0.3,gamma=2.0):
    a=np.asarray(class_logits,float);v=np.asarray(mask_logits,float);lp=a-logsumexp(a,axis=1,keepdims=True)
    costs=np.empty((len(a),len(targets)))
    for i in range(len(a)):
        for j,t in enumerate(targets):
            # Ordinary task has one label and completely known masks.
            y=np.maximum(np.asarray(t['mask'],float),0)
            costs[i,j]=-lp[i,t['classes'][0]]+np.mean(np.logaddexp(0,v[i])-y*v[i])
    if len(targets):
        rr,cc=linear_sum_assignment(costs);pairs=sorted(zip(cc.tolist(),rr.tolist()))
    else:pairs=[]
    used={i for j,i in pairs}
    total=sum(costs[i,j] for j,i in pairs)-beta*sum(lp[i,0] for i in range(len(a)) if i not in used)
    return {'loss':float(total),'assignment':[i for j,i in pairs]}
