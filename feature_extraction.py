import os
import sys
import time
import math
import pickle
import subprocess
import json

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import cv2 as cv
import mediapipe as mp
import seaborn as sn
import smogn
import statsmodels.api as sm

from numpy.fft import fft, ifft
from statistics import mode, median, quantiles

from scipy.signal import find_peaks

#Helper functions
def distance(x0,y0,x1,y1):
    '''
    Distance in cartesian coordinate system
    '''
    return math.sqrt((x0-x1)**2+(y0-y1)**2)

def angular_amplitude_entropy(values, min_val=0, max_val=90, n_buckets=18):
    '''
    Given a series of angular amplitudes (degree), find the entropy of the series.
    Assumptions: the values are between 0 and 90
    '''
    dA = (max_val - min_val) / (n_buckets - 1)
    buckets = np.arange(min_val, max_val + 1, dA)
    n = np.histogram(values, buckets)[0]
    p = n / n.sum()
    p[p == 0] = 1
    lp = np.log(p)
    ppe = -np.multiply(p, lp).sum() / np.log(n_buckets)
    return ppe

def period_entropy(values, min_val=0, max_val=2, n_buckets=50):
    '''
    Given a series of periods (s), find the entropy of the series.
    Assumptions: the values are between 0s and 2s
    '''
    dA = (max_val - min_val) / (n_buckets - 1)
    buckets = np.arange(min_val, max_val + 1, dA)
    n = np.histogram(values, buckets)[0]
    p = n / n.sum()
    p[p == 0] = 1
    lp = np.log(p)
    ppe = -np.multiply(p, lp).sum() / np.log(n_buckets)
    return ppe

def entropy(p):
    '''
    p: np.array of probabilities
    '''
    return -(p*np.log(p)).sum()

def get_length(filename):
    '''
    Given a video filename, find its length in seconds.
    '''
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

# In[41]:


def denoise(D, WINDOW_SIZE = 3, THRESHOLD = 3):
    '''
    Denoise (missing hands) a series of angular distances. 
    Look at values before and after. If majority are missing hands, most likely the task is yet to start, 
    and this is actually a missing hand. If minority are missing, do interpolation to replace the missing value.
    Parameters:
            WINDOW_SIZE: how many time-steps we are looking at to the left and right of 
            the current time-step when hand is not detected by mediapipe
            
            THRESHOLD: if number neighbor frames with undetected hands is at most the threshold, then interpolate
    '''
    
    '''
    Learn a 3-degree polynomial Y that copies D or <fit and interpolate> D for missing data
    '''
    D = list(D)
    Y = []
    for i in range(0,len(D)):
        if D[i] != -1.0:
            Y.append(D[i])
        else:
            Y.append(np.nan)
    
    Y = pd.Series(Y)
    Y = Y.interpolate(method="polynomial", order=3)
    
    for i in range(0,len(Y)):
        if Y[i]<0:
            Y[i] = -1.0
    
    '''
    Look at <WINDOW_SIZE> values left and right to the current one. 
    If missing values > <THRESHOLD>, it remains a missing value. Otherwise, interpolation value (Y) is used.
    '''
    
    for i in range(0,len(D)):
        if D[i]==-1.0:
            if i>=WINDOW_SIZE:
                vals_before = D[(i-WINDOW_SIZE):(i-1)]
            else:
                vals_before = D[0:(i-1)]
                for j in range(0,WINDOW_SIZE-len(vals_before)):
                    vals_before = [-1.0] + vals_before
                    
            if i<(len(D)-WINDOW_SIZE):
                vals_after = D[(i+1):(i+WINDOW_SIZE)]
            else:
                vals_after = D[(i+1):]
                for j in range(0,WINDOW_SIZE-len(vals_after)):
                    vals_after = vals_after + [-1.0]
    
            vals = vals_before + [-1.0] + vals_after
            if(len(np.argwhere(np.asarray(vals)==-1.0))<=THRESHOLD):
                D[i] = Y[i]
                if np.isnan(D[i]):
                    D[i] = -1.0
    
    return np.asarray(D)


# In[42]:


def custom_peaks(D, distance):
    '''
    Given a signal D(t), determine the peaks with the constraint: 
        peak-to-peak distance must be at least 'distance' (expressed as the number of frames)
    Method: First, find the peaks with distance constraint. 
            Then determine the peaks from the middle part of the time series, take the maximum peak.
            If maximum peak>30, set maximum peak to 30 (might be due to noise or projection issue)
            Add extra constraint: any detected peak must have height at least one-third of the maximum peak
            Find peaks again from the input time series with height and distance constraints.
    '''
    peaks, _ = find_peaks(D, distance=(int)(distance))
    n_peaks = len(peaks)
    middle_peaks = D[peaks[(int)(np.floor((n_peaks-1)/4)):(int)(np.floor(3*(n_peaks-1)/4))]]
    high_peak = np.percentile(middle_peaks, 80)
    height = np.floor(high_peak/2)
    #print("Peak min height: %.3f"%(height))
    #max_height = np.minimum(np.max(middle_peaks),30)
    #height = np.floor(max_height/3)
    peaks, _ = find_peaks(D, distance=(int)(distance), height = (int)(height))
    return peaks


# In[43]:


def custom_bottoms(D, distance):
    #ensure this is a numpy array
    D = np.asarray(list(D))
    DI = 180 - D
    return custom_peaks(DI, distance)


# In[44]:


def get_stats(series):
    '''
    series = [periods_denoised, periods_trimmed]
    '''
    stats = {}
    stats['median'] = median(series)
    stats['quartile_range'] = iqr(series)
    stats['mean'] = np.mean(series)
    stats['min'] = np.min(series)
    stats['max'] = np.max(series)
    stats['stdev'] = np.std(series)

    return stats

def linear_regression_fit(series_x, series_y):
    fit = {}
    series_x = np.asarray(series_x).reshape((-1, 1))
    series_y = np.asarray(series_y)
    model = LinearRegression()
    model.fit(series_x, series_y)
    fit["fitness_r2"] = model.score(series_x, series_y)
    fit["slope"] = model.coef_[0]
    return fit
                               
def degree_for_good_fit(series_x, series_y, fitness_threshold = 0.90):
    d = 0
    r2 = 0
    while r2<fitness_threshold:
        d +=1
        x = np.asarray(series_x)
        y = np.asarray(series_y)
        z = np.polyfit(x, y, d)
        p = np.poly1d(z)
        r2 = r2_score(y,p(x))
        
        if d>=10:
            return d
        
    return d

# Customized on top of MediaPipe output to detect the correct hand and track key points
class HandTrackerCustomized():
    '''
    Use mediapipe to track coordinates of finger joints
    '''
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        '''
        Default initialization, maxHands is set to 2 since both hands are visible in some videos
        '''
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,hand='left',draw=True):
        '''
        Finf the coordinates for a specific hand (left/right)
        inputs:
            image (BGR; video frame)
            hand: left, right
            draw: draw coordinates on the image?
        '''
        hand = hand.lower()
        
        imageRGB = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        landmarks = {}
        
        SINGLE_HANDED = False
        '''
        Mediapipe may detect multiple hands. 
        We need to identify the mediapipe index of the specific hand we are trying to identify.
        '''
        LH_INDEX = -1
        
        '''
        Assuming Mediapipe is predicting the reverse hand.
        The reversal is necessary because mediapipe expects the selfie videos as mirrored. 
        But PARK platforms does not apply mirror effect.
        Need to consider this if any future change impacts this assumption.
        '''
        
        t_label = {"left":"Right","right":"Left"}
        
        if self.results.multi_handedness:
            if len(self.results.multi_handedness)==1:
                '''
                Mediapipe detects a single hand. 
                Just verify that it is our desired hand (L/R), set the index to 0.
                Else, the index remains -1, means no hand detected.
                '''
                SINGLE_HANDED = True
                which_hand = self.results.multi_handedness[0].classification[0].label
                conf_score = self.results.multi_handedness[0].classification[0].score
                
                if which_hand==t_label[hand] and conf_score>0.9:
                    LH_INDEX = 0
                
            else:
                '''
                Mediapipe detects multiple hands.
                Find out the desired hand and set the index accordingly. If only one hand, all good!
                If none of the hands is the desired one, the index remains -1.
                If two desired hands (most likely, multiple people in the camera), select the bigger hand.
                If more than two desired hand, this is an exceptional case we need to re-implement. For now, return undetected hand (-1).
                '''
                indexes = []
                
                for j in range(0,len(self.results.multi_handedness)):
                    if self.results.multi_handedness[j].classification[0].label==t_label[hand] and self.results.multi_handedness[j].classification[0].score>0.9:
                        indexes.append(j)
                        
                if len(indexes)==1:
                    LH_INDEX = indexes[0]
                    
                elif len(indexes)==2:
                    #Calculate size and pick the bigger one
                    print("Both hands detected as %s\n"%(hand))
                    handLms0 = self.results.multi_hand_landmarks[indexes[0]]
                    handLms1 = self.results.multi_hand_landmarks[indexes[1]]
        
                    WRIST0 = [handLms0.landmark[0].x, handLms0.landmark[0].y, handLms0.landmark[0].z] 
                    THUMB0 = [handLms0.landmark[4].x, handLms0.landmark[4].y, handLms0.landmark[4].z]
                    d0 = distance(WRIST0[0],WRIST0[1], THUMB0[0], THUMB0[1])
                
                    WRIST1 = [handLms1.landmark[0].x, handLms1.landmark[0].y, handLms1.landmark[0].z] 
                    THUMB1 = [handLms1.landmark[4].x, handLms1.landmark[4].y, handLms1.landmark[4].z]
                    d1 = distance(WRIST1[0],WRIST1[1], THUMB1[0], THUMB1[1])
                    
                    if d0>d1:
                        LH_INDEX = indexes[0]
                    else:
                        LH_INDEX = indexes[1]
    
                elif len(indexes)>2:
                    print("EXCEPTION #1: More than two %s hands found in the video.\n"%(hand))
                    print("="*20)
                    
        
        '''
        If the hand is not detected, landmark remains empty {}.
        '''
        if LH_INDEX == -1:
            return image, landmarks
        
        '''
        Now that we know the index of the desired hand, just extract the landmarks that matter and return.
        '''
        handLms = self.results.multi_hand_landmarks[LH_INDEX]
        
        landmarks["WRIST"] = [handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[0].z] 
        landmarks["THUMB_TIP"] = [handLms.landmark[4].x, handLms.landmark[4].y, handLms.landmark[4].z]
        landmarks["INDEX_FINGER_TIP"] = [handLms.landmark[8].x, handLms.landmark[8].y, handLms.landmark[8].z]
        landmarks["MIDDLE_FINGER_TIP"] = [handLms.landmark[12].x, handLms.landmark[12].y, handLms.landmark[12].z]
        landmarks["RING_FINGER_TIP"] = [handLms.landmark[16].x, handLms.landmark[16].y, handLms.landmark[16].z]
        landmarks["PINKY_TIP"] = [handLms.landmark[20].x, handLms.landmark[20].y, handLms.landmark[20].z]
        landmarks["THUMB_CMC"] = [handLms.landmark[1].x, handLms.landmark[1].y, handLms.landmark[1].z]

        if draw:
            self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                
        return image, landmarks

class Signal:
    
    NOT_FOUND = -1
    INTERRUPTION_SPEED_THRESHOLD = 50 #degree per second
    INTERRUPTION_MIN_DURATION = 0.20 #second
    FREEZE_SPEED_THRESHOLD = 50 #degree per second
    FREEZE_MIN_DURATION = 0.30 #second
        
    def __init__(self, raw, wrist_raw, num_frames, duration):
        self.raw_signal = raw
        self.wrist_raw = wrist_raw
        self.raw_fft = fft(self.raw_signal)
        self.NUM_FRAMES = num_frames
        self.DURATION = duration
        self.PER_FRAME_DURATION = self.DURATION/self.NUM_FRAMES
        self.denoised_signal, self.wrist_denoised = self.interpolation_and_denoise()
        
        self.peaks_denoised = self.peak_detection(self.denoised_signal)
        self.peaks_trimmed = np.asarray(self.peaks_denoised[1:-1]-self.peaks_denoised[1])
        
        self.trimmed_signal = np.asarray(self.denoised_signal[self.peaks_denoised[1]:(self.peaks_denoised[-2]+1)])
        self.wrist_trimmed = self.wrist_denoised[self.peaks_denoised[1]:(self.peaks_denoised[-2]+1)]
        
        self.signals = {'r':self.raw_signal, 'd':self.denoised_signal, 't':self.trimmed_signal}
        self.peaks = {'d':self.peaks_denoised, 't':self.peaks_trimmed}
        self.periods_denoised = []
        self.periods_trimmed = []
        self.speed_denoised = []
        self.speed_trimmed = []
        self.acceleration_denoised = []
        self.acceleration_trimmed = []
        
        for i in range(1,len(self.peaks_denoised)):
            self.periods_denoised.append((self.peaks_denoised[i]-self.peaks_denoised[i-1])*self.PER_FRAME_DURATION)
            
        for i in range(1,len(self.peaks_trimmed)):
            self.periods_trimmed.append((self.peaks_trimmed[i]-self.peaks_trimmed[i-1])*self.PER_FRAME_DURATION)
        
        for i in range(0,len(self.denoised_signal)-1):
            self.speed_denoised.append(self.denoised_signal[i+1]-self.denoised_signal[i])
            
        self.speed_denoised = np.asarray(self.speed_denoised) #degree per frame
        self.speed_denoised = self.speed_denoised/self.PER_FRAME_DURATION #degree per second
        
        for i in range(0,len(self.trimmed_signal)-1):
            self.speed_trimmed.append(self.trimmed_signal[i+1]-self.trimmed_signal[i])
            
        self.speed_trimmed = np.asarray(self.speed_trimmed) #degree per frame
        self.speed_trimmed = self.speed_trimmed/self.PER_FRAME_DURATION #degree per second
        self.speeds = {'d':self.speed_denoised, 't':self.speed_trimmed}
        
        for i in range(0,len(self.speed_denoised)-1):
            self.acceleration_denoised.append(self.speed_denoised[i+1]-self.speed_denoised[i])
            
        self.acceleration_denoised = np.asarray(self.acceleration_denoised) #degree per frame*second
        self.acceleration_denoised = self.acceleration_denoised/self.PER_FRAME_DURATION #degree per second2
        
        for i in range(0,len(self.speed_trimmed)-1):
            self.acceleration_trimmed.append(self.speed_trimmed[i+1]-self.speed_trimmed[i])
            
        self.acceleration_trimmed = np.asarray(self.acceleration_trimmed) #degree per frame*second
        self.acceleration_trimmed = self.acceleration_trimmed/self.PER_FRAME_DURATION #degree per second2

    def interpolation_and_denoise(self):
        D = denoise(self.raw_signal)
        
        '''
        Take the maximum length segment where angle is not -1 (valid angular distance -- visible hand)
        '''
        first_frame = self.NOT_FOUND
        last_frame = self.NOT_FOUND

        max_first_frame = -1
        max_last_frame = -1
        max_num_frames = 0


        for i in range(0,len(D)):
            if D[i]!=self.NOT_FOUND:
                if first_frame==self.NOT_FOUND:
                    first_frame = i
                else:
                    last_frame = i
                    num_frames = last_frame - first_frame +1
                    if num_frames>max_num_frames:
                        max_num_frames = num_frames
                        max_first_frame = first_frame
                        max_last_frame = last_frame
            else:
                first_frame = self.NOT_FOUND
                last_frame = self.NOT_FOUND

        D = D[max_first_frame:max_last_frame+1]
        W = self.wrist_raw[max_first_frame:max_last_frame+1]
        return D, W
    
    def peak_detection(self, D):
        
        '''
        Run peak detection, the parameters could be further improved/optimized
        '''
        X = np.arange(0,len(D))
        MIN_PERIOD = 0.15 #in seconds
        d_min = (int)(MIN_PERIOD/self.PER_FRAME_DURATION)
        peaks = custom_peaks(D, distance=d_min)
    
        '''
        Cancel peaks where there is no minima between this peak and the previous peak
        '''
        n_peaks = len(peaks)
        
        bottoms = custom_bottoms(D, d_min)
        n_bottoms = len(bottoms)
        middle_bottoms = D[bottoms[(int)(np.floor((n_bottoms-1)/4)):(int)(np.floor(3*(n_bottoms-1)/4))]]
        BOTTOM_MAX_HEIGHT = 10
        print("Number of frames for a peak: %d"%(d_min))
        print("Bottom max height: %.3f"%(BOTTOM_MAX_HEIGHT))
    
        peaks_denoised = [peaks[0]]
        for i in range(0,len(peaks)-1):

            min_val = 180.0
            for j in range(peaks[i],peaks[i+1]):
                if D[j]<min_val:
                    min_val = D[j]

            if min_val<BOTTOM_MAX_HEIGHT:
                peaks_denoised.append(peaks[i+1])
            
        return peaks_denoised
        
    def aperiodicity(self,signal_version):
        '''
        signal_version = ['r','d','t']
        '''
        X = fft(self.signals[signal_version.lower()])
        power_spectrum = np.square(np.abs(X))
        power_spectrum = power_spectrum/power_spectrum.sum()
        return entropy(power_spectrum)
        
    def interruption_count(self,signal_version):
        '''
        signal version = ['d','t']
        '''
        n = 0
        S = np.abs(self.speeds[signal_version.lower()])
        t = 0
        for i in range(0,len(S)):
            if S[i]<=self.INTERRUPTION_SPEED_THRESHOLD:
                t +=1
            else:
                if (t*self.PER_FRAME_DURATION)>=self.INTERRUPTION_MIN_DURATION:
                    n +=1
                t = 0
        return n
    
    def freeze_count(self,signal_version):
        '''
        signal version = ['d','t']
        '''
        n = 0
        S = np.abs(self.speeds[signal_version.lower()])
        t = 0
        for i in range(0,len(S)):
            if S[i]<=self.FREEZE_SPEED_THRESHOLD:
                t +=1
            else:
                if (t*self.PER_FRAME_DURATION)>=self.FREEZE_MIN_DURATION:
                    n +=1
                t = 0
        return n
    
    def max_freeze_duration(self,signal_version):
        '''
        signal version = ['d','t']
        '''
        S = np.abs(self.speeds[signal_version.lower()])
        t = 0
        t_max = 0
        for i in range(0,len(S)):
            if S[i]<=self.FREEZE_SPEED_THRESHOLD:
                t +=1
            else:
                if t>t_max:
                    t_max = t
                t = 0
        return (t_max*self.PER_FRAME_DURATION)
    
    def amplitude_decrement(self,signal_version):
        '''
        signal version = ['d','t']
        '''
        D = self.signals[signal_version]
        t = self.peaks[signal_version]
        A = D[t]
        n = len(A)
        assert (n>=2),"Not enough peaks to analyze"
        n1 = round(n/2)
        feats = linear_regression_fit(t, -A)
        feats['end_to_mean'] = np.mean(A) - A[-1]
        feats['fit_min_degree'] = degree_for_good_fit(t,A)
        feats['last_to_first_half'] = np.mean(A[:(n1+1)]) - np.mean(A[(n1+1):])
            
        return feats
    
    def amplitude_stats(self, signal_version):
        '''
        signal version = ['d','t']
        '''
        D = self.signals[signal_version]
        t = self.peaks[signal_version]
        A = D[t]
        texts = {"d":"denoised", "t":"trimmed"}
        feats = {}
        amp_stats = get_stats(A)
        for k in amp_stats.keys():
            feats["amplitude_"+k+"_"+texts[signal_version]] = amp_stats[k]
            
        feats['amplitude_entropy_'+texts[signal_version]] = angular_amplitude_entropy(A)
        
        return feats
        
    
    def wrist_movements(self):
        W = self.wrist_trimmed
        n = len(W)
        movements_x = []
        movements_y = []
        movements_d = []
        for i in range(1,n):
            (x1,y1) = W[i]
            (x0,y0) = W[i-1]
            
            if x1==self.NOT_FOUND or x0==self.NOT_FOUND:
                movements_x.append(0)
                movements_y.append(0)
                movements_d.append(0)
            else:
                movements_x.append((x1-x0)/self.PER_FRAME_DURATION)
                movements_y.append((y1-y0)/self.PER_FRAME_DURATION)
                movements_d.append((distance(x0,y0,x1,y1))/self.PER_FRAME_DURATION)
                
        feats = {}
        feats_x = get_stats(np.abs(movements_x))
        for k in feats_x.keys():
            feats['wrist_mvmnt_x_'+k] = feats_x[k]
            
        feats_y = get_stats(np.abs(movements_y))
        for k in feats_y.keys():
            feats['wrist_mvmnt_y_'+k] = feats_y[k]
            
        feats_d = get_stats(np.abs(movements_d))
        for k in feats_d.keys():
            feats['wrist_mvmnt_dist_'+k] = feats_d[k]
        
        return feats

#Feature extractor from MediaPipe Key Points
def get_final_features(data):
    '''
    data: 
    {
        'D_raw':np.array, 
        'W_raw': list of normalized wrist coordinates (x,y),
        'num_frames':int,
        'duration':float
    }
    '''
    
    signal = Signal(data['D_raw'], data['W_raw'], data['num_frames'], data['duration'])
    
    '''
    Features related to wrist movement (horizontal, vertical, and in cartesian coord)
    '''
    features = signal.wrist_movements()
    
    '''
    Features related to rhythm
    '''
    features['aperiodicity_denoised'] = signal.aperiodicity('d')
    features['aperiodicity_trimmed'] = signal.aperiodicity('t')
    features['periodEntropy_denoised'] = period_entropy(signal.periods_denoised)
    features['periodEntropy_trimmed'] = period_entropy(signal.periods_trimmed)
    features['periodVarianceNorm_denoised'] = np.var(signal.periods_denoised)/np.max(signal.periods_denoised)
    features['periodVarianceNorm_trimmed'] = np.var(signal.periods_trimmed)/np.max(signal.periods_trimmed)
    features['numInterruptions_denoised'] = signal.interruption_count('d')
    features['numInterruptions_trimmed'] = signal.interruption_count('t')
    features['numFreeze_denoised'] = signal.freeze_count('d')
    features['numFreeze_trimmed'] = signal.freeze_count('t')
    features['maxFreezeDuration_denoised'] = signal.max_freeze_duration('d')
    features['maxFreezeDuration_trimmed'] = signal.max_freeze_duration('t')
    
    '''
    Statistics of Period, Frequency, and Amplitude
    '''
    period_stats_denoised = get_stats(signal.periods_denoised)
    for k in period_stats_denoised.keys():
        features['period_'+k+"_denoised"] = period_stats_denoised[k]
         
    period_stats_trimmed = get_stats(signal.periods_trimmed)
    for k in period_stats_trimmed.keys():
        features['period_'+k+"_trimmed"] = period_stats_trimmed[k]
        
    features['period_entropy_denoised'] = period_entropy(signal.periods_denoised)
    features['period_entropy_trimmed'] = period_entropy(signal.periods_trimmed)
    
    frequency_stats_denoised = get_stats(1.0/np.asarray(signal.periods_denoised))
    for k in frequency_stats_denoised.keys():
        features['frequency_'+k+"_denoised"] = frequency_stats_denoised[k]
        
    frequency_stats_trimmed = get_stats(1.0/np.asarray(signal.periods_trimmed))
    for k in frequency_stats_trimmed.keys():
        features['frequency_'+k+"_trimmed"] = frequency_stats_trimmed[k]
        
    frequency_fit_denoised = linear_regression_fit(np.arange(0,len(signal.periods_denoised)), 1.0/np.asarray(signal.periods_denoised))
    for k in frequency_fit_denoised.keys():
        features['frequency_lr_'+k+'_denoised'] = frequency_fit_denoised[k]
        
    frequency_fit_trimmed = linear_regression_fit(np.arange(0,len(signal.periods_trimmed)), 1.0/np.asarray(signal.periods_trimmed))
    for k in frequency_fit_trimmed.keys():
        features['frequency_lr_'+k+'_trimmed'] = frequency_fit_trimmed[k]
        
    features['frequency_fit_min_degree_denoised'] = degree_for_good_fit(np.arange(0,len(signal.periods_denoised)), 1.0/np.asarray(signal.periods_denoised))
    features['frequency_fit_min_degree_trimmed'] = degree_for_good_fit(np.arange(0,len(signal.periods_trimmed)), 1.0/np.asarray(signal.periods_trimmed))
    
    amp_stats = signal.amplitude_stats('d')
    for k in amp_stats:
        features[k] = amp_stats[k]
        
    amp_stats = signal.amplitude_stats('t')
    for k in amp_stats:
        features[k] = amp_stats[k]
    '''
    Amplitude decrement
    '''
    amp_dec_denoised = signal.amplitude_decrement('d')
    for k in amp_dec_denoised.keys():
        features['amplitude_decrement_'+k+'_denoised'] = amp_dec_denoised[k]
        
    amp_dec_trimmed = signal.amplitude_decrement('t')
    for k in amp_dec_trimmed.keys():
        features['amplitude_decrement_'+k+'_trimmed'] = amp_dec_trimmed[k]
        
    '''
    Signal
    '''
    features['num_peaks_trimmed'] = len(signal.peaks_trimmed)
    features['num_peaks_denoised'] = len(signal.peaks_denoised)
    features['num_interruptions_norm_denoised'] = features['numInterruptions_denoised']/features['num_peaks_denoised']
    features['num_freeze_norm_denoised'] = features['numFreeze_denoised']/features['num_peaks_denoised']
    features['num_interruptions_norm_trimmed'] = features['numInterruptions_trimmed']/features['num_peaks_trimmed']
    features['num_freeze_norm_trimmed'] = features['numFreeze_trimmed']/features['num_peaks_trimmed']
    
    speed_stats_denoised = get_stats(np.abs(signal.speed_denoised))
    for k in speed_stats_denoised.keys():
        features['speed_'+k+"_denoised"] = speed_stats_denoised[k]
        
    speed_stats_trimmed = get_stats(np.abs(signal.speed_trimmed))
    for k in speed_stats_trimmed.keys():
        features['speed_'+k+"_trimmed"] = speed_stats_trimmed[k]
        
    acceleration_stats_denoised = get_stats(np.abs(signal.acceleration_denoised))
    for k in acceleration_stats_denoised.keys():
        features['acceleration_'+k+"_denoised"] = acceleration_stats_denoised[k]
        
    acceleration_stats_trimmed = get_stats(np.abs(signal.acceleration_trimmed))
    for k in acceleration_stats_trimmed.keys():
        features['acceleration_'+k+"_trimmed"] = acceleration_stats_trimmed[k]
        
    return features


#Extract features from a given file and specified target hand
def extract_features(filename, output_path, hand, labels=(0,"")):
    '''
    For the filename, create a folder and save the plots, MP annotations, and mid-level features there.
    return features as a dictionary
    
    hand: left, right
    '''
    
    '''
    <intermediate-feature-file>  
    <output-video-with-mp-annotation>
    '''
    
    hand = hand.lower()
    FEATURE_DIR = output_path
    annotations = {}
    (annotations['rating'], annotations['diagnosis']) = labels
    features = {}
    
    base_file = os.path.basename(filename)
    base_file = base_file[0:base_file.find(".webm")]
    
    full_dir_path = os.path.join(FEATURE_DIR,base_file)
    if not os.path.exists(full_dir_path):
        os.mkdir(full_dir_path)
    
    full_dir_path = os.path.join(full_dir_path,hand.upper())
    
    if os.path.exists(full_dir_path):
        with open(os.path.join(full_dir_path,"intermediate_features.pkl"), 'rb') as handle:
            data = pickle.load(handle)
            features = get_final_features(data)
    
        return features
    
    '''
    Process a new file
    '''
    os.mkdir(full_dir_path)
    annotation_image_path = os.path.join(full_dir_path,"MP Annotation Frames")
    annotation_json_path = os.path.join(full_dir_path,"MP Annotation JSON")
    os.mkdir(annotation_image_path)
    os.mkdir(annotation_json_path)
    
    cap = cv.VideoCapture(filename)
    
    tracker = HandTrackerCustomized()
    cv.startWindowThread()
    
    D = [] #Time-series of angular distance
    W = [] #Time-seris of wrist coordinates (normalized): (W.x, W.y) -- requires all videos to be in the same size
    NOT_FOUND = -1 #Constant
    NUM_FRAMES = 0 #Count the number of frames in the video
    
    while(1):
        ret, frame = cap.read()
        if not ret:
            break

        NUM_FRAMES +=1

        width  = (int)(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = (int)(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        #print(width, height)
        
        if NUM_FRAMES==1:
            out = cv.VideoWriter(os.path.join(full_dir_path,"output.mp4"), cv.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

        #frame is updated by mediapipe (ref call)
        visual, landmarks = tracker.handsFinder(frame, hand=hand, draw=True)
        
        '''
        Below is the data structure of landmarks returned:
            landmarks["WRIST"] = [x y z]
            landmarks["THUMB_TIP"] = [x y z]
            landmarks["INDEX_FINGER_TIP"] = [x y z]
            landmarks["MIDDLE_FINGER_TIP"] = [x y z]
            landmarks["RING_FINGER_TIP"] = [x y z]
            landmarks["PINKY_TIP"] = [x y z]
        '''
        
        if "WRIST" in landmarks.keys():
            wrist_x = (int)(landmarks["WRIST"][0]*width)
            wrist_y = (int)(landmarks["WRIST"][1]*height)

            thumb_x = (int)(landmarks["THUMB_TIP"][0]*width)
            thumb_y = (int)(landmarks["THUMB_TIP"][1]*height)

            index_x = (int)(landmarks["INDEX_FINGER_TIP"][0]*width)
            index_y = (int)(landmarks["INDEX_FINGER_TIP"][1]*height)

            cv.line(frame, (wrist_x, wrist_y), (thumb_x, thumb_y), (255,0,0), 2)
            cv.line(frame, (wrist_x, wrist_y), (index_x, index_y), (0,0,255), 2)

            Vector_WT = ((thumb_x-wrist_x),(thumb_y-wrist_y))
            Vector_WI = ((index_x-wrist_x),(index_y-wrist_y))
            dot = Vector_WT[0]*Vector_WI[0] + Vector_WT[1]*Vector_WI[1]
            cosx = dot/(math.sqrt((Vector_WT[0]**2)+(Vector_WT[1]**2))*math.sqrt((Vector_WI[0]**2)+(Vector_WI[1]**2)))
            cosx = np.minimum(cosx,1.0)
            angle = (math.acos(cosx)*180)/math.pi

            D.append(angle)
            
            thumb_cmc_x = (int)(landmarks["THUMB_CMC"][0]*width)
            thumb_cmc_y = (int)(landmarks["THUMB_CMC"][1]*height)
            
            w_norm = distance(wrist_x, wrist_y, thumb_cmc_x, thumb_cmc_y)
            W.append((wrist_x/w_norm, wrist_y/w_norm))
            
        else:
            D.append(NOT_FOUND)
            W.append((NOT_FOUND, NOT_FOUND))

        
        cv.imshow(filename, frame)
        
        out.write(frame)
        
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cv.destroyAllWindows()
    cv.waitKey(1)
    out.release()
    
    '''
    Extract intermediate representations for the features
    '''
    
    INT_FEATS = {}
    INT_FEATS["D_raw"] = np.asarray(D)
    INT_FEATS["W_raw"] = W
    
    #Video Processing Done
    DURATION = get_length(filename)
            
    PER_FRAME_DURATION = DURATION/NUM_FRAMES
    
    INT_FEATS["duration"] = DURATION
    INT_FEATS["num_frames"] = NUM_FRAMES
    
    with open(os.path.join(full_dir_path,"intermediate_features.pkl"), 'wb') as handle:
        pickle.dump(INT_FEATS, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    with open(os.path.join(full_dir_path,"intermediate_features.pkl"), 'rb') as handle:
        data = pickle.load(handle)

        features = get_final_features(data)

    return features