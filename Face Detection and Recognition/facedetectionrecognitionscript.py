import paho.mqtt.client as mqtt 
import time 
import numpy as np
from PIL import Image 
import tflite_runtime.interpreter as tflite 

import os 
import cv2 

cap = cv2.VideoCapture(0)


model_dir = "models/custom" 

model = "custom_detection_model.tflite"

label = "custom_labels.txt"

mqttClient = mqtt.Client("greenhouse_alarm")
mqttClient.connect('127.0.0.1', 1883)
mqttClient.loop_start()

model_path=os.path.join(model_dir,model)
label_path=os.path.join(model_dir,label)

def detect_objects(interpreter, image, score_threshold=0.3, top_k=6):

    set_input_tensor(interpreter, image)
    invoke_interpreter(interpreter)
    
    global model_dir
    scores = get_output_tensor(interpreter, 0)
    boxes = get_output_tensor(interpreter, 1)
    #count = int(get_output_tensor(interpreter, 2))
    class_ids = get_output_tensor(interpreter, 3)
  
    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            box=Box(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

import collections
Object = collections.namedtuple('Object', ['id', 'score', 'box'])

class Box(collections.namedtuple('Box', ['xmin', 'ymin', 'xmax', 'ymax'])):
    __slots__ = ()

import re
def load_labels(path):
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels
      
def make_interpreter(path):
    interpreter = tflite.Interpreter(model_path=path)

    print('Loading Model: {} '.format(path))
    
    return interpreter

def input_image_size(interpreter):
    """Returns input image size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels
    
def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  image = image.resize((input_image_size(interpreter)[0:2]), resample=Image.NEAREST)

    
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]

  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def invoke_interpreter(interpreter):
  global inference_time_ms
  
  t1=time.time()
  interpreter.invoke()
  inference_time_ms = (time.time() - t1) * 1000
  print("****Inference time = ", inference_time_ms)
  
def overlay_text_detection(objs, labels, cv2_im, fps):
    height, width, channels = cv2_im.shape
    font=cv2.FONT_HERSHEY_SIMPLEX
  
    for obj in objs:
        x0, y0, x1, y1 = list(obj.box)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        box_color, text_color, thickness=(255,0,0), (0,0,0),1
            
        text3 = '{}'.format(labels.get(obj.id, obj.id))
        print(text3)

        msg = text3
        info = mqttClient.publish(
            topic='iot/test',
            payload=msg.encode('utf-8'),
            qos=0,
        )

        info.wait_for_publish()
        print(info.is_published())
        
        try:
          cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), box_color, thickness)
          cv2_im = cv2.rectangle(cv2_im, (x0,y1-10), (x1, y1+10), (255,255,255), -1)
          cv2_im = cv2.putText(cv2_im, text3, (x0, y1),font, 0.6, text_color, thickness)
        except:
          #log_error()
          pass
    
    global model, inference_time_ms
    str1="FPS: " + str(fps)
    cv2_im = cv2.putText(cv2_im, str1, (width-180, height-55),font, 0.7, (255, 0, 0), 2)
    
    str2="Inference: " + str(round(inference_time_ms,1)) + " ms"
    cv2_im = cv2.putText(cv2_im, str2, (width-240, height-25),font, 0.7, (255, 0, 0), 2)
    
    cv2_im = cv2.rectangle(cv2_im, (0,height-20), (width, height), (0,0,0), -1)
    cv2_im = cv2.putText(cv2_im, model, (10, height-5),font, 0.6, (0, 255, 0), 2)
    
    return cv2_im

def main():

  interpreter = make_interpreter(model_path)
  
  interpreter.allocate_tensors()
  
  labels = load_labels(label_path)
  
  fps=1

  while True:
        
        start_time=time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2_im_rgb)
       
        
        results = detect_objects(interpreter, image)
        cv2_im = overlay_text_detection(results, labels, cv2_im, fps)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        cv2.imshow('Detect Objects', cv2_im)
        
        elapsed_ms = (time.time() - start_time) * 1000
        fps=round(1000/elapsed_ms,1)
        print("--------fps: ",fps,"---------------")
        
if __name__ == '__main__':
  main()
