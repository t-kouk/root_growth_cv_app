# Image analysisis of two images
# Identify roots from binary images and collect found root tip informatation as location coordinates and length 
# (= diameter) into dictionary
# User can adjust how big are the tips they are searching.

# import functions 
from root_growth_analyzer.root_tips.fileWITHfunctions import to_csv, combine_DF, color_image, color_image_v2, aux_analysis_1, aux_analysis_2, main_analysis, create_file_list, check_date, find_contours, processImage, draw_circles_around, add_Text, process_ML_image, extract_data_DF, compareTips, check_file, extract_date

import pandas as pd
import sys
import os
import cv2
import glob
from datetime import datetime
import numpy as np
import logging

# KOMENTORIVIESIMERKKI: 
#  python scenario2.py hydescan1_T001_L001_2020.07.15_033029_362_DYY-prediction.jpg  hydescan1_T001_L001_2020.07.16_033029_363_DYY-prediction.jpg hydescan1_T001_L001_2020.07.16_033029_363_DYY.jpg 10  

""" 

STEP 1: Check how many day should be analyzed

STEP 2 OPTION 1: 
If time period is 2 days --> read images as unchanged 1-channel gray image and convert them to binary
Find contours and collect root tip information into the dictionaries 
Create dataframes for further comparison. 
1. Check root tip diameters: diameter of combined image should be more or same.
2. Calculate distance between centre points, if it's less than 30 then root tips from IMAGE #1 and
combined image are same. 
3. Create a dataframe with root tips that have been found from IMAGE #1 and combine image (IMAGE #1 and IMAGE #2) 
Draw tips that exist in IMAGE #1 and in combined image
Save dataframe to csv. Create image with root tips. 

STEP 3 OPTION 2:
Time period is more than 2 days--> read first two images as unchanged 1-channel gray image and convert them to binary
Find contours and collect root tip information into the dictionaries 
Create dataframes for further comparison. 
1. Check root tip diameters: diameter of combined image should be more or same.
2. Calculate distance between centre points, if it's less than 30 then root tips from IMAGE #1 and
combined image are same. 
3. Create a dataframe with root tips that have been found from IMAGE #1 and combine image (IMAGE #1 and IMAGE #2) 
Draw tips that exist in IMAGE #1 and in combined image

Repeat same actions for all image exapt for last image. Last image get tip information from first day dataframe

Save dataframe to csv. Create image with found root tips
"""

def root_tip_analysis(im1_s, im2_s, im2_o, tip_size, results_dir, date=None):
    tip_size = int(tip_size)/0.1    # 1 pixel = 0,1 mm

    # Calculate how many days/images should be analyzed
    period = check_date(im1_s, im2_s, im2_o)

    # Period is less than 1 --> mistake
    if period <1:
        logging.error("Invalid dates in filenames, period is less than 1")
        return None

    # Period is 1 --> simple analysis of two images, OPTION 1
    elif period == 1:
        logging.debug("Analyze day #1 and day #2")
        _, _, _, result_DF = main_analysis(im1_s, im2_s, tip_size)
        if result_DF is None:
            return None
        root_tip_images = f'{results_dir}/root_tip_growth_images'
        if not os.path.exists(root_tip_images):
            os.mkdir(root_tip_images)
        root_tip_results = f'{results_dir}/root_tip_growth_changes'
        if not os.path.exists(root_tip_results):
            os.mkdir(root_tip_results)
        # Create color image with found root tips
        im_original = color_image(im2_o, result_DF)

        # Prepare dafaframe for writing in csv
        result_DF = to_csv(result_DF)

        # Save dataframe to csv-file
        result_DF.to_csv(f'{root_tip_results}/ROOT_TIP_CHANGES_{date}.csv', sep=';', index=False, float_format='%.1f')

        # Save processed image
        cv2.imwrite(f'{root_tip_images}/ROOT_TIP_CHANGES_{date}.png', im_original)

    # Period is more than 1 day --> OPTION 2
    else:
        # NOTE: Currently app doesn't support calling this, there seems to be bugs
        logging.info(f"Looking for tips from {period} days")
        seg_files = '/'.join(im1_s.split('/')[:-1]) + '/'
        image_list = create_file_list(im1_s, period, seg_files)

        # Analyze firs pair of images and get combined image for further analysis and temporary dataframe
        # nex_day_image is binary combine image of day #1 and #2
        combined_image_b, combined_image_g, starting_df, temp_df = main_analysis(image_list[0], image_list[1], tip_size)
        # Counter for image #3 and forward
        counter = 1
        while counter < len(image_list):
            combined_image_b, combined_image_g, temp_df = aux_analysis_1(combined_image_b, combined_image_g, image_list[counter], tip_size, temp_df)
            counter += 1
            if counter == len(image_list)-1:
                logging.info("Counter is %s, %s", counter, image_list[counter])
                # Call the second versio of image analysis since this is last file
                result_DF = aux_analysis_2(combined_image_b, combined_image_g, image_list[counter], tip_size, temp_df, im2_o, starting_df)

                # Draw found root tips in original o
                im_original = color_image_v2(im2_o, starting_df, result_DF)

                # Prepare dafaframe for writing in csv
                result_DF = to_csv(result_DF)

                result_DF['Filename_Start'] = im1_s.split('/')[-1]
                result_DF['Filename_Last'] = im2_s.split('/')[-1]

                # Combine information from starting dataframe with result dataframe
                result_DF = combine_DF(starting_df, result_DF)

                # Save dataframe to csv-file
                filelabel = f'{results_dir}/ROOT_TIPS'
                logging.info('Saving to %s', results_dir)
                result_DF.to_csv(f'{filelabel}.csv', sep=';', index=False, float_format='%.1f')

                # Save processed image
                cv2.imwrite(f'{filelabel}.png', im_original)
                break
            else:
                pass

    return result_DF


if __name__ == "__main__":

    if len(sys.argv)==5:
        # First image is processed image after root segmentation, IMAGE #1
        im1_s = sys.argv[1]

        # First image is processed image after root segmentation, IMAGE #2
        # Second image is original image for root tip drawing, IMAGE #2
        im2_s = sys.argv[2]
        im2_o = sys.argv[3]

        # Size of the root tip for analysis1.py 
        # 1 pixel = 0,1 mm
        tip_size = int(sys.argv[4])/0.1

        # Folders with images
        seg_files = '20210428_root_growth_cv-main/DL_use_model/data/prediction_result/'
        orig_files = '20210428_root_growth_cv-main/DL_use_model/data/prediction_data/'

        # Create correct pathnames
        im1_s = seg_files + im1_s
        im2_s = seg_files + im2_s
        im2_o = orig_files + im2_o

        if tip_size>1:
            print(f'Searching tips with diameter over: {int(tip_size)}px')
            main(im1_s, im2_s, im2_o, tip_size)
        else:
            print("Root tip size is too small")
    else:
        print("Check filenames or tip size")