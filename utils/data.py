import cv2
import glob
import numpy as np
import os

def aly_parse_res(root_dir):
    res = sorted(glob.glob(os.path.join(root_dir, 'raw_parse/*.png'))) 
    
    for ele in res:
        ele_res = cv2.imread(ele, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        cls_res = np.unique(ele_res)
        print(np.where(ele_res==4))


def merge_cls():
    # cls_map: parse results
    atts =   ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    new_atts = ['skin', 'brow', 'brow', 'eye', 'eye', 'eye_g', 'ear', 'ear', 'ignore', 
            'nose', 'mouth', 'lip', 'lip',  'neck',  'ignore', 'ignore', 'hair', 'ignore']
    # 19-> 11
    new_map = {
        'skin':1, 
        'brow':2, 
        'eye':3, 
        'eye_g':4, 
        'ear':5,   
        'nose':6, 
        'mouth':7, 
        'lip':8, 
        'neck':9,  
        'hair':10, 
        'ignore': 11,
    }

    ids_map = {}
    for i, (att, new_att) in enumerate(zip(atts, new_atts), 1):
        print(i, att, new_att)
        ids_map[i] = new_map[new_att]
    print(ids_map)
    return ids_map
    

if __name__ == "__main__":
    import sys
    # aly_parse_res(sys.argv[1])
    a = (np.random.random((1, 10))*20).astype(int)
    ids_map = merge_cls()
    print(a)
    for ids in ids_map:
        a[a==int(ids)] = int(ids_map[ids])
    print(a)