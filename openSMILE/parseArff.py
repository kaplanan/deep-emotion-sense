import arff
import numpy as np

rowCount = 0

f_arff = open("f_arff.txt", "w")

for row in arff.load('RECOLA.arff'):
    rowCount += 1

arr = np.zeros((rowCount, 14))
id_arr = np.zeros((rowCount, 1))

i = 0
for row in arff.load('RECOLA.arff'):
    arr[i] = row.pcm_fftMag_mfcc[0]
    arr[i] = row.pcm_fftMag_mfcc[1]
    arr[i] = row.pcm_fftMag_mfcc[2]
    arr[i] = row.pcm_fftMag_mfcc[3]
    arr[i] = row.pcm_fftMag_mfcc[4]
    arr[i] = row.pcm_fftMag_mfcc[5]
    arr[i] = row.pcm_fftMag_mfcc[6]
    arr[i] = row.pcm_fftMag_mfcc[7]
    arr[i] = row.pcm_fftMag_mfcc[8]
    arr[i] = row.pcm_fftMag_mfcc[9]
    arr[i] = row.pcm_fftMag_mfcc[10]
    arr[i] = row.pcm_fftMag_mfcc[11]
    arr[i] = row.pcm_fftMag_mfcc[12]
    i += 1
    f_arff.write(arr[i] + "\n")
