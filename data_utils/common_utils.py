import math
import numpy as np
from PIL import Image

def trans2_white_bg(img_path):
    img = Image.open(img_path)
    img_arr = 255 - np.array(img)[:, :, 3]
    img_ = Image.fromarray(img_arr)
    img_.save(img_path)
    return img_arr

def affine_shear(seq, dx=-0.3, dy=0.0):
    mask = ~(seq == 0)
    seq_12 = seq.copy()
    seq_12[:,4] -= 12.0
    seq_12[:,5]  = -seq_12[:,5] + 12
    seq_12[:,6] -= 12.0
    seq_12[:,7]  = -seq_12[:,7] + 12
    seq_12[:,8] -= 12.0
    seq_12[:,9]  = -seq_12[:,9] + 12
    
    seq_args = seq_12[:,4:]
    seq_args = np.concatenate([seq_args[:, :2], seq_args[:, 2:4], seq_args[:, 4:6]], 0).transpose(1,0)
    affine_matrix=np.array([[1, dx], 
                            [dy, 1]])
    rotated_args = np.dot(affine_matrix,seq_args)
    rotated_args = rotated_args.transpose(1,0)
    new_args = np.concatenate([rotated_args[:71], rotated_args[71:142], rotated_args[142:]],-1)
    new_args[:,0] += 12.0
    new_args[:,1] = -(new_args[:,1] - 12)
    new_args[:,2] += 12.0
    new_args[:,3] = -(new_args[:,3] - 12)
    new_args[:,4] += 12.0
    new_args[:,5] = -(new_args[:,5] - 12)
    new_seq = np.concatenate([seq[:, :4], new_args],1)
    new_seq = new_seq * mask
    return new_seq

def affine_scale(seq, scale=0.8):
    mask = ~(seq==0)
    seq_args = seq[:, 4:] - 12.0
    seq_args *= scale
    seq_args = seq_args + 12.0
    new_seq = np.concatenate([seq[:, :4], seq_args], 1)
    new_seq = new_seq * mask
    return new_seq

def affine_rotate(seq,theta=-5):
    mask = ~(seq==0)
    seq_12 = seq.copy()
    seq_12[:,4] -=12.0
    seq_12[:,5]  = -seq_12[:,5] + 12
    seq_12[:,6] -=12.0
    seq_12[:,7]  = -seq_12[:,7] + 12
    seq_12[:,8] -=12.0
    seq_12[:,9]  = -seq_12[:,9] + 12
    
    seq_args =seq_12[:, 4:] # default as [71,6]
    seq_args = np.concatenate([seq_args[:,:2],seq_args[:,2:4],seq_args[:,4:6]],0).transpose(1,0)# note 2,213
    theta = math.radians(theta)
    affine_matrix=np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta), np.cos(theta)]])# note 2,2
    rotated_args = np.dot(affine_matrix,seq_args)# note 2,213
    rotated_args = rotated_args.transpose(1,0)# note 213,2
    new_args = np.concatenate([rotated_args[:71],rotated_args[71:142],rotated_args[142:]],-1)# note 2,213
    new_args[:,0] +=12.0
    new_args[:,1] = -(new_args[:,1]-12)
    new_args[:,2] +=12.0
    new_args[:,3] = -(new_args[:,3]-12)
    new_args[:,4] +=12.0
    new_args[:,5] = -(new_args[:,5]-12)
    
    new_seq = np.concatenate([seq[:,:4],new_args],1)
    new_seq =new_seq *mask
    return new_seq

