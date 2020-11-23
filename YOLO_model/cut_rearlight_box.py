from darknet import performDetect, performDetect_bin, performDetect_mask, performDetect_gaussian, performDetect_kmeans

from bbox_cluster import compute_centroids_for_bboxes

import os
from PIL import Image
import numpy as np


def check_folder():
    original_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/valid'
    restore_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/YOLO_dataset_one_class/valid'
    
    #class_table_list = ['BLO','BLR','BOO','BOR','OLO','OLR','OOO','OOR']
    ori_alllist = os.listdir(original_dir)
    for folder in ori_alllist:
        class_res_folder = os.path.join(restore_dir, folder)
        if(not os.path.exists(class_res_folder)):
            print('create new folder')
            os.makedirs(class_res_folder)
        class_ori_folder = os.path.join(original_dir, folder)
        for frames_folder in os.listdir(class_ori_folder):
            ori_frames_folder = os.path.join(class_ori_folder, frames_folder)
            res_frames_folder = os.path.join(class_res_folder, frames_folder)
            #print(res_frames_folder)
            if(not os.path.exists(res_frames_folder)):
                print('create new folder: frames')
                os.makedirs(res_frames_folder)
            
            for img_name in os.listdir(ori_frames_folder):
                new_img_name = os.path.join(res_frames_folder, img_name)
                if(os.path.exists(new_img_name)):
                    print('skip')
                    continue
                ori_img_name_path = os.path.join(ori_frames_folder, img_name)
                #print(ori_img_name_path)
                detections = performDetect(imagePath=ori_img_name_path, thresh= 0.25, configPath = "yolov4-taillight.cfg", weightPath = "/home/huxi/YOLO_v4/darknet/backup/yolov4-taillight_2000.weights", metaPath= "data/taillights.data")
                img_with_boxes = detections['image']
                img = Image.fromarray(img_with_boxes, 'RGB')

                #print(new_img_name)
                img.save(new_img_name)

def cut_img():
    original_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/valid'
    restore_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/YOLO_cut_dataset/valid'
    
    #class_table_list = ['BLO','BLR','BOO','BOR','OLO','OLR','OOO','OOR']
    ori_alllist = os.listdir(original_dir)
    for folder in ori_alllist:
        class_res_folder = os.path.join(restore_dir, folder)
        if(not os.path.exists(class_res_folder)):
            print('create new folder')
            os.makedirs(class_res_folder)
        class_ori_folder = os.path.join(original_dir, folder)
        for frames_folder in os.listdir(class_ori_folder):
            ori_frames_folder = os.path.join(class_ori_folder, frames_folder)
            res_frames_folder = os.path.join(class_res_folder, frames_folder)
            #print(res_frames_folder)
            if(not os.path.exists(res_frames_folder)):
                print('create new folder: frames')
                os.makedirs(res_frames_folder)
            
            for img_name in os.listdir(ori_frames_folder):
                new_img_name = os.path.join(res_frames_folder, img_name)
                if(os.path.exists(new_img_name)):
                    print('skip')
                    continue
                ori_img_name_path = os.path.join(ori_frames_folder, img_name)
                #print(ori_img_name_path)
                detections = performDetect_bin(imagePath=ori_img_name_path, thresh= 0.10, configPath = "yolov4-taillight.cfg", weightPath = "/home/huxi/YOLO_v4/darknet/backup/yolov4-taillight_2000.weights", metaPath= "data/taillights.data")
                
                img_with_boxes = detections['image']
                img = Image.fromarray(img_with_boxes, 'RGB')
                img.save(new_img_name)

def generate_mask():
    original_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/valid'
    restore_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/YOLO_mask_dataset/valid'

    ori_alllist = os.listdir(original_dir)
    for folder in ori_alllist:
        class_res_folder = os.path.join(restore_dir, folder)
        if(not os.path.exists(class_res_folder)):
            print('create new folder')
            os.makedirs(class_res_folder)
        class_ori_folder = os.path.join(original_dir, folder)
        for frames_folder in os.listdir(class_ori_folder):
            ori_frames_folder = os.path.join(class_ori_folder, frames_folder)
            res_frames_folder = os.path.join(class_res_folder, frames_folder)
            #print(res_frames_folder)
            if(not os.path.exists(res_frames_folder)):
                print('create new folder: frames')
                os.makedirs(res_frames_folder)
            
            for img_name in os.listdir(ori_frames_folder):
                new_img_name = os.path.join(res_frames_folder, img_name)
                if(os.path.exists(new_img_name)):
                    print('skip')
                    continue
                ori_img_name_path = os.path.join(ori_frames_folder, img_name)
                #print(ori_img_name_path)
                detections = performDetect_mask(imagePath=ori_img_name_path, thresh= 0.10, configPath = "yolov4-taillight.cfg", weightPath = "/home/huxi/YOLO_v4/darknet/backup/yolov4-taillight_2000.weights", metaPath= "data/taillights.data")
                
                mask = detections['mask']
                #np.save(new_img_name[-4:] + '.npy', mask)
                img = Image.fromarray(mask, 'RGB')
                img.save(new_img_name)


def generate_mask_gaussian():
    original_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/train'
    restore_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/YOLO_mask_gaussian_dataset/train'

    ori_alllist = os.listdir(original_dir)
    for folder in ori_alllist:
        class_res_folder = os.path.join(restore_dir, folder)
        if(not os.path.exists(class_res_folder)):
            print('create new folder')
            os.makedirs(class_res_folder)
        class_ori_folder = os.path.join(original_dir, folder)
        for frames_folder in os.listdir(class_ori_folder):
            ori_frames_folder = os.path.join(class_ori_folder, frames_folder)
            res_frames_folder = os.path.join(class_res_folder, frames_folder)
            #print(res_frames_folder)
            if(not os.path.exists(res_frames_folder)):
                print('create new folder: frames')
                os.makedirs(res_frames_folder)
            
            for img_name in os.listdir(ori_frames_folder):
                new_img_name = os.path.join(res_frames_folder, img_name)
                if(os.path.exists(new_img_name)):
                    print('skip')
                    continue
                ori_img_name_path = os.path.join(ori_frames_folder, img_name)
                #print(ori_img_name_path)
                detections = performDetect_gaussian(imagePath=ori_img_name_path, thresh= 0.10, configPath = "yolov4-taillight.cfg", weightPath = "/home/huxi/YOLO_v4/darknet/backup/yolov4-taillight_2000.weights", metaPath= "data/taillights.data")
                
                mask = detections['mask']
                #np.save(new_img_name[-4:] + '.npy', mask)
                img = Image.fromarray(mask, 'RGB')
                img.save(new_img_name)
                img.close()

def kmeans_generate_mask():
    original_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/dataset/train'
    restore_dir = '/media/huxi/DATA/inf_master/Semester-5/Thesis/code/YOLO_mask_clustered_dataset/train'
    ori_alllist = os.listdir(original_dir)

    for folder in ori_alllist:
        class_res_folder = os.path.join(restore_dir, folder)
        if(not os.path.exists(class_res_folder)):
            print('create new folder')
            os.makedirs(class_res_folder)
        class_ori_folder = os.path.join(original_dir, folder)
        for frames_folder in os.listdir(class_ori_folder):
            ori_frames_folder = os.path.join(class_ori_folder, frames_folder)
            res_frames_folder = os.path.join(class_res_folder, frames_folder)
            #print(res_frames_folder)
            if(not os.path.exists(res_frames_folder)):
                print('create new folder: frames')
                os.makedirs(res_frames_folder)

            all_bbox = []
            all_size = []
            new_img_name = os.path.join(res_frames_folder, 'mask_image.png')
            if(os.path.exists(new_img_name)):
                print('skip')
                continue

            for img_name in os.listdir(ori_frames_folder):
                #new_img_name = os.path.join(res_frames_folder, img_name)

                ori_img_name_path = os.path.join(ori_frames_folder, img_name)
                #print(ori_img_name_path)
                detections = performDetect_kmeans(imagePath=ori_img_name_path, thresh= 0.10, configPath = "yolov4-taillight.cfg", weightPath = "/home/huxi/YOLO_v4/darknet/backup/yolov4-taillight_2000.weights", metaPath= "data/taillights.data")
                
                boundingBox = detections['boundingBox']
                image_size = detections['image_size']
                image = detections['image']

                all_bbox.extend(boundingBox)
                all_size.append(image_size)
            all_size = np.array(all_size)
            centroids = compute_centroids_for_bboxes(all_bbox, 3)
            image_size_mean = np.mean(all_size, axis=0)
            mask = np.ones((int(image_size_mean[0]), int(image_size_mean[1]), 3), dtype=np.uint8)
            for centroid in centroids:
                up_left_coord = [ int((centroid.x - centroid.w/2)*image_size_mean[1]),  int((centroid.y - centroid.h/2)*image_size_mean[0])]
                down_right_coord = [ int((centroid.x + centroid.w/2)*image_size_mean[1]),  int((centroid.y + centroid.h/2)*image_size_mean[0])]

                mask[max(up_left_coord[1] - 10, 0): min(down_right_coord[1] + 10, int(image_size_mean[0])), max(up_left_coord[0] - 10, 0): min(down_right_coord[0] + 10, int(image_size_mean[1])), :] = 255
            img = Image.fromarray(mask, 'RGB')
            img.save(new_img_name)
            img.close()

if __name__ == '__main__':
    #check_folder()
    #cut_img()
    #generate_mask()
    #generate_mask_gaussian()
    kmeans_generate_mask()









