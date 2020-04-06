# MultimodalDiarization
Multimodal speaker diarization using a pre-trained audio-visual synchronization model

## Requirements:
- SyncNet's pre-trained model is required which can be downloaded from https://github.com/joonson/syncnet_python. Required python files from SyncNet project is already uploaded in this repository. 
- Some important required packages are: __pyannote.metrics__, __pyannote.core__, __face_recognition__, __librosa__.
- Data set can be downloaded from http://groups.inf.ed.ac.uk/ami/download/. The data set requires MixHeadset recordings and four closeup camera recordings. 

## How to run
- Main file is **Multimodal_ReDiarization_Syncnet_FaceDetection_AMI.py** You need to set some paths in this file. Additionally set the recording name to variable **AudioDataSet = IS1008a** and set closeup camera recordig to **video = Closeup1.avi**. Run!
- You need to run this code for each closeup recording (Closeup1, Closeup2, Closeup3 and Closeup4). 
- Finally run the last couple of line of the code by making flag **FindDER = True** to acquire diarization error rate (DER). 

## Cite
- Please cite this paper if you use this code. <br />  
**Ahmad, R.; Zubair, S.; Alquhayz, H.; Ditta, A. Multimodal Speaker Diarization Using a Pre-Trained Audio-Visual Synchronization Model. Sensors 2019, 19, 5163.**
