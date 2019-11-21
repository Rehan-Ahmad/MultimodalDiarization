# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:48:15 2018

@author: Rehan
"""
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.diarization import DiarizationPurity
import face_recognition
import cv2
import time
import scipy.stats.mstats as stats
import numpy as np
from gmm import *
from SAD import silenceRemoval
from pyannote.core import Segment, Timeline, Annotation
import librosa
import xml.etree.ElementTree as ET
from copy import copy
from sklearn.preprocessing import StandardScaler
import pickle
import pdb, subprocess,os
from operator import itemgetter
from itertools import groupby
from SyncNetInstance import *
from scipy import signal
from Mutimodal_ReDiarization_AudioResegmentation import AudioResegmentationGMM

def write_to_RTTM(rttm_file_name, sp_file_name,\
                  meeting_name, most_likely, \
                  seg_length, total_num_frames):

    print("...Writing out RTTM file...")
    #do majority voting in chunks of 250
    duration = seg_length
    chunk = 0
    end_chunk = duration

    max_gmm_list = []

    smoothed_most_likely = np.array([], dtype=np.float32)

    while end_chunk < len(most_likely):
        chunk_arr = most_likely[list(range(chunk, end_chunk))]
        max_gmm = stats.mode(chunk_arr)[0][0]
        max_gmm_list.append(max_gmm)
        smoothed_most_likely = np.append(smoothed_most_likely, max_gmm*np.ones(seg_length)) #changed ones from 250 to seg_length
        chunk += duration
        end_chunk += duration

    end_chunk -= duration
    if end_chunk < len(most_likely):
        chunk_arr = most_likely[list(range(end_chunk, len(most_likely)))]
        max_gmm = stats.mode(chunk_arr)[0][0]
        max_gmm_list.append(max_gmm)
        smoothed_most_likely = np.append(smoothed_most_likely,\
                                         max_gmm*np.ones(len(most_likely)-end_chunk))
    most_likely = smoothed_most_likely
    
    out_file = open(rttm_file_name, 'w')

    with_non_speech = -1*np.ones(total_num_frames)

    if sp_file_name:
        speech_seg = np.loadtxt(sp_file_name, delimiter=' ',usecols=(0,1))
        speech_seg_i = np.round(speech_seg).astype('int32')
#            speech_seg_i = np.round(speech_seg*100).astype('int32')
        sizes = np.diff(speech_seg_i)
    
        sizes = sizes.reshape(sizes.size)
        offsets = np.cumsum(sizes)
        offsets = np.hstack((0, offsets[0:-1]))

        offsets += np.array(list(range(len(offsets))))
    
    #populate the array with speech clusters
        speech_index = 0
        counter = 0
        for pair in speech_seg_i:
            st = pair[0]
            en = pair[1]
            speech_index = offsets[counter]
            
            counter+=1
            idx = 0
            for x in range(st+1, en+1):
                with_non_speech[x] = most_likely[speech_index+idx]
                idx += 1
    else:
        with_non_speech = most_likely
        
    cnum = with_non_speech[0]
    cst  = 0
    cen  = 0
    for i in range(1,total_num_frames): 
        if with_non_speech[i] != cnum: 
            if (cnum >= 0):
                start_secs = ((cst)*0.01)
                dur_secs = (cen - cst + 2)*0.01
#                    out_file.write("SPEAKER " + meeting_name + " 1 " +\
#                                   str(start_secs) + " "+ str(dur_secs) +\
#                                   " <NA> <NA> " + "speaker_" + str(cnum) + " <NA>\n")
                out_file.write("SPEAKER " + meeting_name + " 1 " +\
                               str(start_secs) + " "+ str(dur_secs) +\
                               " speaker_" + str(cnum) + "\n")
            cst = i
            cen = i
            cnum = with_non_speech[i]
        else:
            cen+=1
              
    if cst < cen:
        cnum = with_non_speech[total_num_frames-1]
        if(cnum >= 0):
            start_secs = ((cst+1)*0.01)
            dur_secs = (cen - cst + 1)*0.01
#                out_file.write("SPEAKER " + meeting_name + " 1 " +\
#                               str(start_secs) + " "+ str(dur_secs) +\
#                               " <NA> <NA> " + "speaker_" + str(cnum) + " <NA>\n")
            out_file.write("SPEAKER " + meeting_name + " 1 " +\
                           str(start_secs) + " "+ str(dur_secs) +\
                           " speaker_" + str(cnum) + "\n")

    print("DONE writing RTTM file")

def SADError(segments, AudioDataSet, annotationlist, audioLength):
    reference = Annotation()
    treeA = ET.parse(annotationlist[0])
    rootA = treeA.getroot()
    for child in rootA.findall('segment'):
        start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
        reference[Segment(start, end)] = 'A'

    treeB = ET.parse(annotationlist[1])
    rootB = treeB.getroot()
    for child in rootB.findall('segment'):
        start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
        reference[Segment(start, end)] = 'A'

    treeC = ET.parse(annotationlist[2])
    rootC = treeC.getroot()
    for child in rootC.findall('segment'):
        start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
        reference[Segment(start, end)] = 'A'

    treeD = ET.parse(annotationlist[3])
    rootD = treeD.getroot()
    for child in rootD.findall('segment'):
        start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
        reference[Segment(start, end)] = 'A'

    hypothesis = Annotation()
    for seg in segments:
        start = seg[0]
        end = seg[1]
        hypothesis[Segment(start, end)] = 'A'
    
    metric = DetectionErrorRate()
    uem = Timeline([Segment(0, audioLength)])
    print('SAD Error Rate: %.2f %%' %(metric(reference, hypothesis, uem=uem)*100))
    
    return metric, reference, hypothesis

def SpeechOnlySamplesOptimal(X,Fs,AudioDataSet, annotationlist):
    # This function makes non-speech (silence + noise) samples to zeros. 
    XSpeech = np.zeros(X.shape)
    treeA = ET.parse(annotationlist[0])
    rootA = treeA.getroot()
    for child in rootA.findall('segment'):
        start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
        XSpeech[int(np.round(start*Fs)):int(np.round(end*Fs))] = copy(X[int(np.round(start*Fs)):int(np.round(end*Fs))])

    treeB = ET.parse(annotationlist[1])
    rootB = treeB.getroot()
    for child in rootB.findall('segment'):
        start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
        XSpeech[int(np.round(start*Fs)):int(np.round(end*Fs))] = copy(X[int(np.round(start*Fs)):int(np.round(end*Fs))])

    treeC = ET.parse(annotationlist[2])
    rootC = treeC.getroot()
    for child in rootC.findall('segment'):
        start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
        XSpeech[int(np.round(start*Fs)):int(np.round(end*Fs))] = copy(X[int(np.round(start*Fs)):int(np.round(end*Fs))])

    treeD = ET.parse(annotationlist[3])
    rootD = treeD.getroot()
    for child in rootD.findall('segment'):
        start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
        XSpeech[int(np.round(start*Fs)):int(np.round(end*Fs))] = copy(X[int(np.round(start*Fs)):int(np.round(end*Fs))])
            
    return XSpeech

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    return iou

def DER(outfile, AudioDataSet,annotationlist, audioLength):
    reference = Annotation()

    if not AudioDataSet=='DiaExample':
        treeA = ET.parse(annotationlist[0])
        rootA = treeA.getroot()
        for child in rootA.findall('segment'):
            start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
            reference[Segment(start, end)] = 'A'
    
        treeB = ET.parse(annotationlist[1])
        rootB = treeB.getroot()
        for child in rootB.findall('segment'):
            start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
            reference[Segment(start, end)] = 'B'
    
        treeC = ET.parse(annotationlist[2])
        rootC = treeC.getroot()
        for child in rootC.findall('segment'):
            start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
            reference[Segment(start, end)] = 'C'
    
        treeD = ET.parse(annotationlist[3])
        rootD = treeD.getroot()
        for child in rootD.findall('segment'):
            start,end = float(child.get('transcriber_start')), float(child.get('transcriber_end'))
            reference[Segment(start, end)] = 'D'
    else:
        reference = Annotation()
        reference[Segment(0.15, 3.41)] = 'A'
        reference[Segment(3.83, 5.82)] = 'A'
        reference[Segment(6.75, 11.10)] = 'B'
        reference[Segment(11.32, 15.8)] = 'C'
        reference[Segment(15.9, 18.8)] = 'B'
        reference[Segment(18.8, 27.8)] = 'C'
        reference[Segment(27.8, 34.4)] = 'B'
        reference[Segment(34.4, 42)] = 'D'

    hypothesis = Annotation()        
    f = open(outfile,'r')
    for line in f.readlines():
        start = float(line.split(' ')[3])
        end = start + float(line.split(' ')[4])
        annotation = line.split(' ')[5][0:-1]
        hypothesis[Segment(start, end)] = annotation
    f.close()
    metric = DiarizationErrorRate()
    metricPurity = DiarizationPurity()
    uem = Timeline([Segment(0, audioLength)])

    print('DER: %.2f %%' %(metric(reference, hypothesis, uem=uem)*100))
    print('Cluster Purity: %.2f %%' %(metricPurity(reference, hypothesis, uem=uem)*100))
    
    return metric, reference, hypothesis

if __name__ == '__main__':
    tic = time.time()
    ###########################################################################
    # Audio Modality
    ###########################################################################
    stWin = 0.03
    stStep = 0.01
    n_mfcc = 19

    AudioDataSet = 'IS1009c' #DiaExample,IS1008a,etc.
    FlagFeatureNormalization = True
    UseSAD = True
    UseSparseFeatures = False
    spnp=None
    SpeechOnlyOptimal = True
    SparseFeatureEngineeringFlag = False
    UseAutoEncoder = True

    fileName = '../data/AMI/'+AudioDataSet+'/audio/'+AudioDataSet+'.Mix-Headset.wav'
    annotationA = '../data/AMI/'+AudioDataSet+'/audio/'+AudioDataSet+'.A.segments.xml'
    annotationB = '../data/AMI/'+AudioDataSet+'/audio/'+AudioDataSet+'.B.segments.xml'
    annotationC = '../data/AMI/'+AudioDataSet+'/audio/'+AudioDataSet+'.C.segments.xml'
    annotationD = '../data/AMI/'+AudioDataSet+'/audio/'+AudioDataSet+'.D.segments.xml'
    annotationlist = [annotationA, annotationB, annotationC, annotationD]
    rttmfile = 'output/'+AudioDataSet+'.rttm'
    if UseSAD: spnp = '../data/AMI/'+AudioDataSet+'/video/'+AudioDataSet+'_spnp.txt'
    meeting_name = AudioDataSet

    # =========================================================================
    # Video data path and output folders.
    # =========================================================================
    video = 'Closeup4.avi'
    videofileName = '../data/AMI/'+AudioDataSet+'/video/'+AudioDataSet+'.'+video
    outputdir = 'output'
    
    if not(os.path.exists(outputdir)):
      os.makedirs(outputdir)
    
    vidfile = os.path.join(outputdir,AudioDataSet+'_i_'+video)
    command = ("ffmpeg -y -i %s -qscale:v 4 -async 1 -r 25 -deinterlace %s"
               % (videofileName,vidfile)) #-async 1
    output = subprocess.call(command, shell=True, stdout=None)

    # Combine audio and Video.
    command = ("ffmpeg -y -i %s -i %s -c copy -shortest %s"
               % (vidfile,fileName,vidfile[0:-4]+'av'+'.avi')) #-async 1
    output = subprocess.call(command, shell=True, stdout=None)
    
    audiofile = os.path.join(outputdir,'MixAudio.wav')
    
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" 
               % (vidfile[0:-4]+'av'+'.avi',audiofile))
    output = subprocess.call(command, shell=True, stdout=None)

    ###########################################################################

    x, Fs = librosa.load(audiofile, sr=16000)
    audioLength = len(x)/(Fs)
    if SpeechOnlyOptimal:
        x = SpeechOnlySamplesOptimal(x,Fs,AudioDataSet, annotationlist)
        
    S = librosa.feature.melspectrogram(y=x, sr=Fs, n_fft=int(Fs*stWin), hop_length=int(Fs*stStep))
    fVects = librosa.feature.mfcc(y=x, S=librosa.power_to_db(S), sr=Fs, n_mfcc = n_mfcc)

    if FlagFeatureNormalization:
        ss = StandardScaler()
        fVects = ss.fit_transform(fVects.T).T
        print("Feature Normalization Done...")

    if UseSAD:
        ###################### Speech Activity Detection ##########################
        segments, idx = silenceRemoval(x, Fs, stWin, stStep, smoothWindow=.0005, Weight=0.0001, plot=False)
        me,re,hy = SADError(segments, AudioDataSet, annotationlist, audioLength)
        ###########################################################################
        
        # Creating a speech/non-speech text File which contains speech only
        # features indices. 'idx' contains speech only features indices.
        st = idx[0]
        newst=copy(st)
        seglist = []
        for i in range(1,idx.size):
            if idx[i]==st+1:
                st+=1
            else:
                en = idx[i-1]
                seglist.append([newst,en])
                st = idx[i]
                newst = copy(st)
        en = idx[i]
        seglist.append([newst,en])
        segarray = np.array(seglist)
        np.savetxt(spnp, segarray, fmt='%d', delimiter=' ')
        #######################################################################
        # Take Speech only frames....
        audioFeatureExamples = fVects.shape[1]
        fVects_speech = copy(fVects[:,idx])
        #######################################################################    
    # =========================================================================
    # Read video frames, do face detection and save into the list.     
    # =========================================================================
    w = 224
    h = 224

    # Load the video and store all the video frames.
    cap = cv2.VideoCapture(vidfile)
    totalVidFrames = int(cap.get(7))-1
    video_frames = []
    facedetectind = []
    framenum = 0
    bslist=[]
    cropxlist=[]
    cropylist=[]
    trbl=[]
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_recognition.face_locations(gray, model='cnn')
            if not faces:
                video_frames.append(np.zeros((w,h,3),dtype=np.uint8))
            else:
                if len(faces)>1:
                    print('len of faces:%d  framenum:%d' %(len(faces),framenum))
                f = faces[0]
                top, right, bottom, left = f
                trbl.append([f,framenum])
                bslist.append(((right-left) + (bottom-top))/4) # (H+W)/4
                cropxlist.append((right+left)/2)  # crop center x
                cropylist.append((bottom + top)/2) # crop center y
#                cv2.rectangle(frame, (left,top),(right,bottom),(255,0,0),2)
#                roi_gray = gray[top:bottom,left:right]
                roi_color = frame[top:bottom,left:right]
#                video_frames.append(cv2.resize(cv2.cvtColor(roi_color,cv2.COLOR_BGR2RGB),(w,h)))
                video_frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                facedetectind.append(framenum)
            framenum = framenum+1
            if framenum%1000 == 0: print('%d frames processed' %framenum);
        else:
            break
    cap.release()
    # =========================================================================
    # Keep Speech only face frames and discard others.
    # =========================================================================
    # Speech only indices for video
    vidind = np.unique(np.array(idx*0.01*25, dtype=int)) 
    # speech + face detected indices only
    vidfaceind = np.intersect1d(vidind, np.array(facedetectind))
    # removing silence+no face video frames. silence indices taken from audio.
    video_frames_speech = [vf for i,vf in enumerate(video_frames) if i in vidfaceind]
    video_frames = video_frames_speech[:]
    del video_frames_speech
    # =============================================================================
    # Shot detection. It finds the continuous face-detected frames and group them.
    # =============================================================================
    shotsvidface = []
    for k,g in groupby(enumerate(vidfaceind), lambda x:x[0]-x[1]):
        shotsvidface.append(list((map(itemgetter(1),g))))

    # compute the length of each shot.
    shotlength = [len(shotsvidface[i]) for i in range(len(shotsvidface))]
    # =========================================================================
    # Apply median filter on box sizes and x,y crop centers to get smooth frames    
    # =========================================================================    
    # removing the non-speech bslist, trbl, cropxlist and cropylist...
    
    trblnew=[]
    bs = []
    cropx = []
    cropy = []
    for i,f in enumerate(trbl):
        if f[1] in vidfaceind:
            trblnew.append(f)
            bs.append(bslist[i])
            cropx.append(cropxlist[i])
            cropy.append(cropylist[i])
    trbl = trblnew[:]
    bslist=bs[:]
    cropxlist=cropx[:]
    cropylist=cropy[:]
    del(trblnew,bs,cropx,cropy)
    ###########################################################################
    trblnew=[]
    bs = []
    cropx = []
    cropy = []
    j=0
    for i,sl in enumerate(shotlength):
        if i==0:
            trblnew.append(trbl[i:sl])
            j=j+sl
        else:
            trblnew.append(trbl[j:j+sl])
            j=j+sl
    for b in trblnew:
        bb=[]
        cx=[]
        cy=[]
        for i,j in enumerate(b):
            bb.append(((j[0][1]-j[0][3]) + (j[0][2]-j[0][0]))/4) # (H+W)/4
            cx.append((j[0][3] + j[0][1])/2) # crop center x
            cy.append((j[0][0] + j[0][2])/2)   # crop center y
        bs.append(bb)
        cropx.append(cx)
        cropy.append(cy)
    
    ###########################################################################
    ind=0
    cs = 0.5
    for i,b in enumerate(bs):
        bsfilt = signal.medfilt(b, kernel_size=min(shotlength)*2-1) #5
        cxfilt = signal.medfilt(cropx[i], kernel_size=min(shotlength)*2-1) #5
        cyfilt = signal.medfilt(cropy[i], kernel_size=min(shotlength)*2-1) #7
        for k in range(len(bsfilt)):
            frame = video_frames[ind]
            bsi = int(bsfilt[k]*(1+2*cs))  # Pad videos by this amount
            frame = np.pad(frame,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(0,0))
            my  = cyfilt[k] + bsi  # BBox center Y
            mx  = cxfilt[k] + bsi  # BBox center X
            video_frames[ind] = cv2.resize(frame[int(my-bsfilt[k]):int(my+bsfilt[k]*(1+2*cs)),
                        int(mx-bsfilt[k]*(1+cs)):int(mx+bsfilt[k]*(1+cs))],(224,224))
            ind = ind + 1
            
    # =========================================================================
    # Save each shot in 2 seconds segments. If shot length is less then 50 
    # frames (2 seconds) and >= 5 frames then save as is. Otherwise discard.
    # =========================================================================
    # Remove all crop videos if they exist. 
    for f in os.listdir(outputdir + '/cropvids/'):
        os.unlink(os.path.join(outputdir, 'cropvids', f))

    vidchunk = 50 # crop video into 50 frames (2sec) chunk.
    print('Cropping each video shot into %dsec segments...'%(vidchunk//25))
    newshots = []
    countervids=0
    counterframes=0
    minvidchunk = 7
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for i,s in enumerate(shotlength):
        if s<=vidchunk and s>=minvidchunk:
            newshots.append(shotsvidface[i])
            out = cv2.VideoWriter(outputdir + '/cropvids/' + str(countervids) + 'crop.avi', fourcc, 25, (w,h))
            for _ in range(s):
                out.write(cv2.cvtColor(video_frames[counterframes], cv2.COLOR_RGB2BGR))
                counterframes=counterframes+1
            out.release()
            countervids=countervids+1
            
        elif s>vidchunk:
            ss = []
            for j in range(0,s-vidchunk,vidchunk):
                out = cv2.VideoWriter(outputdir + '/cropvids/' + str(countervids) + 'crop.avi', fourcc, 25, (224,224))
                for k in range(vidchunk):
                    out.write(cv2.cvtColor(video_frames[counterframes], cv2.COLOR_RGB2BGR))
                    ss.append(shotsvidface[i][k+j])
                    counterframes=counterframes+1
                out.release()
                countervids=countervids+1
                newshots.append(ss)
                ss=[]
            ss=[]
            if j+vidchunk < s and s-j-vidchunk >= minvidchunk:
                out = cv2.VideoWriter(outputdir + '/cropvids/' + str(countervids) + 'crop.avi', fourcc, 25, (224,224))
                for m in range(s-j-vidchunk):
                    out.write(cv2.cvtColor(video_frames[counterframes], cv2.COLOR_RGB2BGR))
                    ss.append(shotsvidface[i][j+vidchunk+m])
                    counterframes=counterframes+1
                out.release()
                countervids=countervids+1
                newshots.append(ss)
            else:
                counterframes=counterframes+s-j-vidchunk                
        else:
            counterframes=counterframes+s
    
    del(video_frames)
    # =========================================================================
    # Crop audio for each newshot and combine with video crop.
    # =========================================================================
    print('Cropping audio and combining with cropped video segments...')
    for i,s in enumerate(newshots):
        audiostart = s[0]/25.0
        audioend = (s[-1]+1)/25.0
        audiotmp = os.path.join(outputdir,'audiotmp.wav')
        # Crop audio file.
        command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss %.3f -to %.3f %s" 
                   % (audiofile,audiostart,audioend,audiotmp))    
        output = subprocess.call(command, shell=True, stdout=None)
        if output != 0:
            pdb.set_trace()
        # Combine audio and video.
        command = ("ffmpeg -y -i %s -i %s -c:v copy -c:a copy %s"
                   % (outputdir + '/cropvids/' + str(i) + 'crop.avi',audiotmp,outputdir + '/cropvids/%04d'%i + '.avi')) #-async 1 
        output = subprocess.call(command, shell=True, stdout=None)
        if output != 0:
            pdb.set_trace()
        os.remove(outputdir + '/cropvids/' + str(i) + 'crop.avi')
    # =========================================================================
    # Run prtrained Syncnet model on each 2 second's video segment. 
    # =========================================================================
    s = SyncNetInstance()
    s.loadParameters('syncnetl2.model')
    print("Model syncnetl2.model loaded.")
    print("Running syncnet on each cropped video...")
    class arguments:
        if not os.path.exists(outputdir+'/syncnet_tmpdir'):
            os.makedirs(outputdir+'/syncnet_tmpdir')
        tmp_dir = os.path.join(outputdir,'syncnet_tmpdir')
        batch_size= 20 #20
        vshift = 15 #15
    opt = arguments()    
    dists = []
    offsets = []
    confs = []

    file = open('f.txt', 'w+')

    for vid in sorted(os.listdir(outputdir + '/cropvids')):
        tic = time.time()
        offset, conf, dist = s.evaluate(opt,videofile=os.path.join(outputdir + '/cropvids',vid))
        toc = time.time()
        file.writelines(vid +' ' +str(toc-tic) + '\n')
        print(toc-tic)
        pdb.set_trace()
        
        offsets.append([offset, vid])
        dists.append(dist)
        confs.append([conf,vid])

    with open(outputdir +'/'+ AudioDataSet + video[0:-4] + '_result.txt', 'w') as fil:
        fil.write('FILENAME\tOFFSET\tCONF\n')
        for ii in range(len(confs)):
          fil.write('%s\t%d\t%.3f\n'%(confs[ii][1], offsets[ii][0], confs[ii][0]))

    print('Total running time: %.2f min' %((time.time()-tic)/60.0))
    # =========================================================================
    # Save newshots as pickle    
    # =========================================================================
    with open('output/'+AudioDataSet+'.'+video[0:-4]+'.shots.pickle', 'wb') as f:
        pickle.dump(newshots,f)

    # =========================================================================
    # Create RTTM file using each saved results and compute DER. 
    # We will only assign high confidence cropped videos only to one speaker.
    # =========================================================================
    FindDER = False
    if FindDER:
#        import pandas as pd
#        allSpeakerShots = []
#        allSpeakerMetric = []
#        for f in os.listdir(outputdir):
#            if f[-6::] == 'pickle':
#                with open(os.path.join(outputdir,f), 'rb') as pf:
#                    allSpeakerShots.append(pickle.load(pf))
#                    print(f)
#            elif f[-3::] == 'txt':
#                allSpeakerMetric.append(pd.read_table(os.path.join(outputdir,f)))
#                print(f)
#
#        out_file = open(os.path.join(outputdir, AudioDataSet + '.rttm'),'w')
#        for j in range(4):
#            for i,sho in enumerate(allSpeakerShots[j]):
#                if allSpeakerMetric[j].iloc[i,1] >=0 and allSpeakerMetric[j].iloc[i,1] <=3:
#                    if allSpeakerMetric[j].iloc[i,2] >= 1:
#                        start = sho[0]
#                        end = sho[-1]
#                        print('clip:%d vid:%d - offset:%d - conf:%f - '
#                              %(j, i, allSpeakerMetric[j].iloc[i,1], allSpeakerMetric[j].iloc[i,2]))
#                        out_file.write("SPEAKER " + AudioDataSet + " 1 " + 
#                                       str(round(start/25.0,3)) + " "+ str(round(len(sho)/25.0,3)) + 
#                                       " speaker_" + str(j) + "\n")
#        out_file.close()
#        print("DONE writing RTTM file")

        # =========================================================================
        # Audio frames assignment using high confidence video frames.
        # =========================================================================
        seg_length = int(150)
        most_likely,_ = AudioResegmentationGMM(fVects,idx, outputdir, AudioDataSet)
        write_to_RTTM(rttmfile, spnp, meeting_name, most_likely, seg_length, fVects.shape[1])
        met, ref, hyp = DER(rttmfile, AudioDataSet, annotationlist,audioLength)

        report = met.report()
        report.to_csv(AudioDataSet+'.csv')
