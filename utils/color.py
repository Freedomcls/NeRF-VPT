import numpy as np
import cv2
import os
# import sys; sys.path.append()


def color_cls(img, pred_map, savedir, prefix="", color_legend=True):
    part_colors = [[255, 0, 0], [255, 0, 255], [255, 170, 0],
                [255, 0, 85], [255, 0, 170],
                [0, 255, 0], [85, 255, 0], [170, 255, 0],
                [0, 255, 85], [0, 255, 170],
                [0, 0, 255], [85, 0, 255], [170, 0, 255],
                [0, 85, 255], [0, 170, 255],
                [255, 255, 0], [255, 255, 85], [255, 255, 170],
                [255, 85, 255], [255, 170, 255],
                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    vis_im = img.copy().astype(np.uint8)
    vis_pred = pred_map.copy().astype(np.uint8)
    vis_pred = cv2.resize(vis_pred, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    vis_pred_color = np.zeros((vis_pred.shape[0], vis_pred.shape[1], 3)) 
    num_cls = np.max(vis_pred) 
    # print(num_cls)
    for i in range(1, num_cls+1):
        index = np.where(vis_pred==i)
        # print(i, index, vis_pred)
        vis_pred_color[index[0], index[1], :] = part_colors[i]
        
    vis_pred_color = vis_pred_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_pred_color, 0.6, 0)

    if color_legend:
        # same with llff_cls dataset
        new_map = {
        'skin':1, 
        'face':2, 
        'neck':3, 
        'head':4,
        'cloth': 5, }
        cls_map = {}
        for key in new_map.keys():
            cls_map[new_map[key]] = key

        start_point = (10, 10)
        delta = 8
        for i in  range(1, num_cls+1):
            color = part_colors[i]
            thickness = -1
            end_point = (start_point[0]+delta, start_point[1]+delta)
            # vis_im = cv2.rectangle(vis_im, start_point, end_point, color, thickness)
            txt_point = (start_point[0]+delta, start_point[1]+delta)
            # vis_im = cv2.putText(vis_im, f"{cls_map.get(i, None)}", txt_point,  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,  cv2.LINE_AA)
            start_point = (start_point[0], start_point[1]+2*delta)
        
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    cv2.imwrite(os.path.join(savedir, prefix+"pred_map.png"), vis_pred_color)
    cv2.imwrite(os.path.join(savedir, prefix+"img_color.png"), vis_im)
    # cv2.imwrite(os.path.join(savedir, prefix+"img_color.png"), cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(savedir, prefix+"img_raw.png"),  cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # test example
    anno = "./bowen_tou/raw_parse/bowen_tou_00025.png"
    img = "./bowen_tou/images/bowen_tou_00025.jpg"
    img = cv2.imread(img)
    anno = cv2.imread(anno)
    color_cls(img, anno, "./debug")
    
