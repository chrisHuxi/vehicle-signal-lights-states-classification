import os

import numpy as np
import cv2 as cv

# reference: https://zhuanlan.zhihu.com/p/82705687
#            https://stackoverflow.com/questions/25074488/how-to-mask-an-image-using-numpy-opencv
def match_SIFT_descriptor(img1, img2):
    # Initiate AKAZE detector
    akaze = cv.AKAZE_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
            
    # Draw matches
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv.imwrite('matches.jpg', img3)
    
    # Select good matched keypoints
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Compute homography
    H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC,5.0)

    # Warp image
    warped_image = cv.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

    
    #cv.imwrite('warped.jpg', warped_image)
    warped_image = cv.resize(warped_image, (img2.shape[1], img2.shape[0]), interpolation = cv.INTER_AREA)

    # compute difference
    difference = cv.subtract(img2, warped_image)
    #hsv_difference = cv.cvtColor(difference, cv.COLOR_BGR2HSV)
    #hsv_difference[:,:,2] += 30
    #difference = cv.cvtColor(hsv_difference, cv.COLOR_HSV2BGR)
    
    mask = np.ones((difference.shape[0], difference.shape[1]),dtype = np.uint8)
    
    mask = np.zeros((difference.shape[0], difference.shape[1]),dtype = np.uint8)
    mask[int(difference.shape[0] * 0.3): int(difference.shape[0] * 0.7), int(difference.shape[1] * 0): int(difference.shape[1] * 0.3)] = 255
    mask[int(difference.shape[0] * 0.3):  int(difference.shape[0] * 0.7), int(difference.shape[1] * 0.7): int(difference.shape[1] * 1.0)] = 255
    #cv.imwrite('mask.jpg', mask) 
    '''
    '''
    difference_1 = cv.bitwise_and(difference,difference,mask = mask)
    return difference_1
    #cv.imwrite('difference.jpg', difference_1)    
    
    
    
if __name__ == '__main__':
    original_dir = u'E:\\inf_master\\Semester-5\\Thesis\\code\\dataset\\valid'
    restore_dir = u'E:\\inf_master\\Semester-5\\Thesis\\code\\SIFT_dataset\\valid'
    
    class_table_list = ['BLO','BLR','BOO','BOR','OLO','OLR','OOO','OOR']
    ori_alllist = os.listdir(original_dir)
    count = 0
    for folder in ori_alllist: #for BLO, ...
        if folder in class_table_list:
            sub_folder = os.path.join(original_dir, folder)
            #print(sub_folder) #E:\inf_master\Semester-5\Thesis\code\dataset\train\BLO
            
            restore_sub_folder = os.path.join(restore_dir, folder) #E:\inf_master\Semester-5\Thesis\code\SIFT_dataset\train\BLO
            
            if(not os.path.exists(restore_sub_folder)):
                print('create new folder')
                os.makedirs(restore_sub_folder)
                
                
            for folder_1 in os.listdir(sub_folder):

                new_folder_addr = os.path.join(restore_sub_folder, folder_1)
                #print(new_folder_addr) #E:\inf_master\Semester-5\Thesis\code\SIFT_dataset\train\OOR\84
                
                if(not os.path.exists(new_folder_addr)):
                    print('create new folder')
                    os.makedirs(new_folder_addr)
                old_folder = os.path.join(sub_folder, folder_1)
                #print(old_folder) #E:\inf_master\Semester-5\Thesis\code\dataset\train\OLO\651
                
                files_list = sorted(os.listdir(old_folder)) #[ 'frame00000695.png', 'frame00000696.png']

                if(len(files_list) - 3 < 10):
                    print('============')
                    print(old_folder)
                    print('============')
                    continue

                for i in range(len(files_list) - 3):
                    image1_name = os.path.join(old_folder, files_list[i]) # E:\inf_master\Semester-5\Thesis\code\dataset\train\BLO\146\frame00001431.png
                    image2_name = os.path.join(old_folder, files_list[i+3])


                    img1 = cv.imread(image1_name)  # referenceImage
                    img2 = cv.imread(image2_name)  # sensedImage
                    new_file_name_with_path = os.path.join(new_folder_addr, files_list[i])
                    print(new_file_name_with_path)
                    try:
                        difference_1 = match_SIFT_descriptor(img2, img1)
                    except:
                        print('smt wrong in match_SIFT_descriptor')
                        continue

                    # print(new_file_name_with_path) #E:\inf_master\Semester-5\Thesis\code\SIFT_dataset\train\BLO\157_not_all\frame00001476.png
                    cv.imwrite(new_file_name_with_path, difference_1) 


        '''
                old_folder = os.path.join(sub_folder, folder_1)

                
                for file in os.listdir(old_folder):

                    old_file_name_with_path = os.path.join(old_folder, file)
                    print(old_file_name_with_path)
                    # \Thesis\code\dataset\train\BLO\108_not_all\frame00004954.png
                    im = Image.open(old_file_name_with_path)
                    im = ImageOps.mirror(im)
                    new_file_name_with_path = os.path.join(new_folder_addr, file)
                    print(new_file_name_with_path)
                    # \Thesis\code\dataset\train\BLO_flip\108_not_all_flip\frame00004954.png
                    im.save(new_file_name_with_path)
                    
                    
    image_dir = u'E:\\inf_master\\Semester-5\\Thesis\\code\\test_image_SIFT'
    files_list = sorted(os.listdir(image_dir))
    print(files_list)
    for i in range(len(files_list) - 3):
        print('========')
        print(files_list[i])
        print(files_list[i+3])
        image1_name = os.path.join(image_dir, files_list[i])
        image2_name = os.path.join(image_dir, files_list[i+3])
    
        img1 = cv.imread(image1_name)  # referenceImage
        img2 = cv.imread(image2_name)  # sensedImage


        difference_1 = match_SIFT_descriptor(img2, img1)
        cv.imwrite( str(i)+ '.png', difference_1) 
    '''    
        