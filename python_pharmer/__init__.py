import os
from nd2reader import ND2Reader
import glob
import numpy as np
from skimage import img_as_bool, measure
from skimage.external.tifffile import imread
import time
import pandas as pd
from scipy import ndimage
from numba import jit
import re



#######################################
#Find Video
def time_info_gather(video_file):
    if video_file is None:	
        video_file=glob.glob('video*')
        if len(video_file) > 1:
            print(30 * '-')
            print("Video File Selection")
            print(30 * '-')
            for i in range(len(video_file)):
                print(i, video_file[i])
			
            choice=input("Enter your choice: ")
            choice=int(choice)
            video_file = video_file[choice]	
    ###########################################################
    # Extract Time Info
    video_data=ND2Reader(video_file)
    
    arr_val2=np.arange(1,len(video_data.timesteps)+1)
    
    foo=np.column_stack((arr_val2, video_data.timesteps))
        
    np.savetxt("time.info.txt",foo,delimiter="\t",header='index\tTime[s]')
    return video_file


def video_roi_extractor(video_file = None):
    #DISCOVER VIDEO INFORMATION
    if video_file is None:
        video_file=glob.glob('$video*nd2$')
    roi_file = glob.glob('roi.1024*')

    #DISCOVER ROI INFORMATION
    rois = imread(roi_file) 
    rois_props = measure.regionprops(rois)
    nregions = len(rois_props)
    print("You have ", nregions, "rois to process")
    
    #INITIALIZE EMPTY FRAMES
    data = {}
    data['ImageNumber'] = []
    data['ObjectNumber'] = []
    data['Intensity_MeanIntensity_f2_340'] = []
    data['Intensity_MeanIntensity_f2_380'] = []
    
    images = ND2Reader(video_file)
    nframes = images.sizes['t']
    print("You have ", nframes, " frames to process")
    start_time = time.time()
    for frame_index in range(nframes):
            if frame_index%100 == 0:
                print(frame_index,"/",nframes, "in", round(time.time()-start_time, 0), " s")
            frame_340 = images.get_frame_2D(t=frame_index, c=0)
            frame_380 = images.get_frame_2D(t=frame_index, c=1)
            regions_340 = measure.regionprops(label_image = rois, intensity_image=frame_340, cache=False)
            regions_380 = measure.regionprops(label_image = rois, intensity_image=frame_380, cache=False)
            for region_index in range(nregions):
                data['ImageNumber'].append(frame_index+1)
                data['ObjectNumber'].append(region_index+1)
                data['Intensity_MeanIntensity_f2_340'].append(regions_340[region_index].mean_intensity)
                data['Intensity_MeanIntensity_f2_380'].append(regions_380[region_index].mean_intensity)
    images.close()
    
    print("--- Reading: %s seconds ---" % (time.time() - start_time))
    df_data = pd.DataFrame(data)
    df_data.to_csv('video_data.txt', header=True, index=False, sep='\t')
    print("Write complete")



#@jit
def video_roi_extractor_faster(video_file = None):
    #DISCOVER VIDEO INFORMATION
    if video_file is None:
        video_find = re.compile('^video.*nd2$')
        video_file = list(filter(video_find.match, os.listdir()))[0]
    roi_file = glob.glob('roi.1024*')

    #DISCOVER ROI INFORMATION
    rois = imread(roi_file) 
    rois_unique = np.setdiff1d( np.unique(rois), [0] )
    nregions = len(rois_unique)
    print("You have ", nregions, "rois to process")
    
    #INITIALIZE EMPTY FRAMES
    ImageNumber = []
    ObjectNumber = []
    f2_340 = []
    f2_380 = []
    
    #LOAD AND PROCESS VIDEO
    images = ND2Reader(video_file)
    nframes = images.sizes['t']
    print("You have ", nframes, " frames to process")
    start_time = time.time()
    #EXTRACT ROI INFORMATION FROM EACH VIDEO FRAMES
    for frame_index in range(nframes):
        if frame_index%100 == 0:
            print(frame_index,"/",nframes, "in", round(time.time()-start_time, 0), " s")
        #KEEP TRACK OF IMAGE NUMBER
        ImageNumber_value = np.repeat(frame_index+1, nregions) 
        ImageNumber.append(ImageNumber_value)
        #KEEP TRACK OF OBJECT NUMBER
        ObjectNumber_value = np.arange(0, nregions)+1
        ObjectNumber.append(ObjectNumber_value)
        #EXTRACT 340 INFO
        frame_340 = images.get_frame_2D(t=frame_index, c=0)
        f2_340_val = ndimage.mean(frame_340, rois, rois_unique)
        f2_340.append(f2_340_val)
        #EXTRACT 380 INFO
        frame_380 = images.get_frame_2D(t=frame_index, c=1)
        f2_380_val = ndimage.mean(frame_380, rois, rois_unique)
        f2_380.append(f2_380_val)        
    images.close()
    
    print("Reading: %s seconds ---" % (time.time() - start_time))
    
    #CONCATENATE THE LISTS MADE DURING THE LOOP TOGETHER
    start_time = time.time()
    ImageNumber = np.concatenate(ImageNumber)
    ObjectNumber = np.concatenate(ObjectNumber)
    f2_340 = np.concatenate(f2_340)
    f2_380 = np.concatenate(f2_380)
    print("concatentation: %s seconds ---" % (time.time() - start_time))
    
    #CREATE DATAFRAME TO EXPORT
    total_data = np.column_stack([ImageNumber, ObjectNumber, f2_340, f2_380 ])
    df_data = pd.DataFrame(total_data)
    df_data.columns = ['ImageNumber', 'ObjectNumber', 'Intensity_MeanIntensity_f2_340', 'Intensity_MeanIntensity_f2_380']
    df_data.to_csv('video_data.txt', header=True, index=False, sep='\t')
    print("Write complete")

# =============================================================================
# os.chdir("Y:/Cris Urcino/190505/190505.63.m.p2 ATP.20uM repeat pulses")
# function_start_timer = time.time()
# video_roi_extractor_faster('video002.nd2')
# function_end_timer = time.time()
# print("It took ", function_end_timer-function_start_timer)
# =============================================================================

