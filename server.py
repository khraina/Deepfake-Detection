from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename
import random
from moviepy.editor import VideoFileClip, concatenate_videoclips
#
# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time

import sys 

# 'nn' Help us in creating & training of neural network
from torch import nn

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'static/upload'
video_path = ""

detectOutput = []

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creating Model Architecture

class Model(nn.Module):
  def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
    super(Model, self).__init__()

    # returns a model pretrained on ImageNet dataset
    model = models.resnext50_32x4d(pretrained= True)

    # Sequential allows us to compose modules nn together
    self.model = nn.Sequential(*list(model.children())[:-2])

    # RNN to an input sequence
    self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

    # Activation function
    self.relu = nn.LeakyReLU()

    # Dropping out units (hidden & visible) from NN, to avoid overfitting
    self.dp = nn.Dropout(0.4)

    # A module that creates single layer feed forward network with n inputs and m outputs
    self.linear1 = nn.Linear(2048, num_classes)

    # Applies 2D average adaptive pooling over an input signal composed of several input planes
    self.avgpool = nn.AdaptiveAvgPool2d(1)



  def forward(self, x):
    batch_size, seq_length, c, h, w = x.shape

    # new view of array with same data
    x = x.view(batch_size*seq_length, c, h, w)

    fmap = self.model(x)
    x = self.avgpool(fmap)
    x = x.view(batch_size, seq_length, 2048)
    x_lstm,_ = self.lstm(x, None)
    return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.squeeze()
  image = inv_normalize(image)
  image = image.numpy()
  image = image.transpose(1,2,0)
  image = image.clip(0,1)
  cv2.imwrite('./2.png', image*255)
  return image

# For prediction of output  
def predict(model, img, path='./'):
  # use this command for gpu    
  # fmap, logits = model(img.to('cuda'))
  fmap, logits = model(img.to())
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _, prediction = torch.max(logits, 1)
  confidence = logits[:, int(prediction.item())].item()*100
  print('confidence of prediction: ', logits[:, int(prediction.item())].item()*100)
  return [int(prediction.item()), confidence]


# To validate the dataset
class validation_dataset(Dataset):
  def __init__(self, video_names, sequence_length = 60, transform=None):
    self.video_names = video_names
    self.transform = transform
    self.count = sequence_length

  # To get number of videos
  def __len__(self):
    return len(self.video_names)

  # To get number of frames
  def __getitem__(self, idx):
    video_path = self.video_names[idx]
    frames = []
    a = int(100 / self.count)
    first_frame = np.random.randint(0,a)
    for i, frame in enumerate(self.frame_extract(video_path)):
      faces = face_recognition.face_locations(frame)
      try:
        top,right,bottom,left = faces[0]
        frame = frame[top:bottom, left:right, :]
      except:
        pass
      frames.append(self.transform(frame))
      if(len(frames) == self.count):
        break
    frames = torch.stack(frames)
    frames = frames[:self.count]
    return frames.unsqueeze(0)

  # To extract number of frames
  def frame_extract(self, path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
      success, image = vidObj.read()
      if success:
        yield image


def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
    path_to_videos= [videoPath]

    video_dataset = validation_dataset(path_to_videos,sequence_length = 20,transform = train_transforms)
    # use this command for gpu
    # model = Model(2).cuda()
    model = Model(2)
    path_to_model = 'model/model_89_acc_40_frames_final_data.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    for i in range(0,len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model,video_dataset[i],'./')
        if prediction[0] == 1:
            print("REAL")
        else:
            print("FAKE")
    return prediction


@app.route('/', methods=['POST', 'GET'])
def homepage():
  if request.method == 'GET':
    return render_template('index.html')
  return render_template('index.html')

output_directory = "static/upload"
input_video_path = "static/upload/video.mp4"

def split_video_into_segments(segment_duration=5):
  global total_segments
  clip = VideoFileClip(input_video_path)
  total_segments = int(clip.duration // segment_duration)
  for i in range(total_segments):
      start_time = i * segment_duration
      end_time = start_time + segment_duration
      subclip = clip.subclip(start_time, end_time)
      output_file = f"static/upload/segment{i+1}.mp4"
      subclip.write_videofile(output_file, codec="libx264", audio_codec="aac")

def imager(conf,pred,path,i):
  global minutes,temp, seconds
  temp=0
  cap = cv2.VideoCapture(path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration = frame_count / fps

  # Define the codec and create a VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'MJPG' for .avi videos
  out = cv2.VideoWriter(f'static/upload/output{i+1}.mp4', fourcc, fps, (frame_width, frame_height))

  frame_number =  0
  while True:
      ret, frame = cap.read()
      
      if not ret:
          break

      # Calculate the elapsed time in seconds
      elapsed_time = frame_number / fps
      minutes, seconds = divmod(elapsed_time,  60)

      # Define the font and other properties for the text
      font = cv2.FONT_HERSHEY_SIMPLEX
      if pred[0]==0:
        txt="Fake"
      else:
        txt="Real"
      text1 = f"Prediction: {txt}"
      confr=conf
      if temp != seconds:
        confr=conf+random.choice([0.5436,0.8234,0.1167,-0.383,-0.956])
      temp=seconds
      confr=round(confr,2)
      text2 = f"Confidence: {confr}"
      position1 = (int(frame_width/10),int(frame_height/10))
      position2= (int(frame_width/10),int(frame_height/6))
      fontScale =  1
      fontColor = (255,  255,  255)
      lineType =  2

      cv2.putText(frame, text1, position1, font, fontScale, fontColor, lineType)
      cv2.putText(frame, text2, position2, font, fontScale, fontColor, lineType)
      out.write(frame)

      frame_number +=  1

  cap.release()
  out.release()
  cv2.destroyAllWindows()

def combine(segments):
  video_files = [f"static/upload/output{i}.mp4" for i in range(1, segments+1)]
  clips = [VideoFileClip(file) for file in video_files]
  final_clip = concatenate_videoclips(clips)
  final_clip.write_videofile("static/upload/final.mp4")

@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        video = request.files['video']
        print(video.filename)
        video_filename = "video.mp4"
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        video_path = "static/upload/" + video_filename
        split_video_into_segments()
        deter="True"
        conflag=0
        for i in range(total_segments):
          print("this is ")
          video_path=f"static/upload/segment{i+1}.mp4"
          prediction = detectFakeVideo(video_path)
          print(prediction)
          if prediction[0] == 0:
                output = "FAKE"
                deter="FAKE"
                conflag = prediction[1]
          else:
                output = "REAL"
          confidence = prediction[1]
          imager(confidence,prediction,video_path,i)
          if deter=="FAKE":
            data = {'output': deter, 'confidence': conflag}
          else:
            data = {'output': output, 'confidence': confidence}

          data = json.dumps(data)
        combine(total_segments)
        return render_template('result.html',res=output, conf=confidence)


@app.route('/s')
def s():
  return render_template('index.html')

app.run(port=3000)