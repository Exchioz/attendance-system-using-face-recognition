from flask import Flask, render_template, Response, request, flash
from keras.models import load_model
import cv2
import os
import pandas as pd
import datetime
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'

socketio = SocketIO(app)

facerecognition_model = "model-cnn-facerecognition.h5"
labels_filename = "labels.csv"
facedetection_model = "haarcascade_frontalface_default.xml"

path = os.path.join(os.getcwd(), os.path.dirname(__file__))
        
if os.path.isfile(os.path.join(path, labels_filename)) == False:
    raise Exception("Can't find %s" % os.path.join(path, labels_filename))
            
labels = pd.read_csv(os.path.join(path, labels_filename))['0'].values
facedetect = cv2.CascadeClassifier(os.path.join(path, facedetection_model))

model = load_model(os.path.join(path, facerecognition_model))

DATASET_PATH = os.path.join(path, "..\dataset")

##curr_frame = None
label_stat = {}
label_count = {}
label_time = {}

for name in labels:
    label_stat[name] = False
    label_count[name] = 0
    label_time[name] = datetime.datetime.now()

def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),  
                  (max(xt, x0 + w), yt), 
                  color, 
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),  
                  (x0 + w, y0 + baseline), 
                  color, 
                  -1)  
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_SIMPLEX,     
                0.5,                          
                text_color,                
                1,
                cv2.LINE_AA) 
    return img

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
            
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (50, 50))
        face_img = face_img.reshape(1, 50, 50, 1)
            
        result = model.predict(face_img)
        idx = result.argmax(axis=1)[0]
        confidence = result.max(axis=1)[0]*100

        if confidence > 80:
            curr_label = labels[idx]
            label_text = "%s (%.2f %%)" % (curr_label, confidence)
            if label_count[curr_label] > 50:
                socketio.emit("prediction", {
                                            ##'frame' :get_curr_frame(),
                                            'label' : curr_label,
                                            'status' : not label_stat[curr_label],
                                            'time' : get_str_datetime()
                })
                socketio.sleep(0.1)
                label_stat[curr_label] = not label_stat[curr_label]
                label_time[curr_label] = datetime.datetime.now()
                label_count[curr_label]= 0

            else :
                if check_diff_time(curr_label):
                    label_count[curr_label] += 1

        else :
            label_text = "N/A"
        frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
    return frame

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                ##curr_frame = frame.copy()
                frame = detect_face(frame)
            except Exception as e:
                print("[ERROR] ", e)
                camera.release()
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

##def get_curr_frame():
        ##frame = cv2.resize(curr_frame, (10,10))
        ##ret, buffer = cv2.imencode('.png', frame)
        ##return buffer.tobytes()

def get_str_datetime():
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def check_diff_time(label):
    timelabel = label_time[label]
    now = datetime.datetime.now()
    
    return now - timelabel > datetime.timedelta(seconds=5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/face_registration")
def face_registration():
    return render_template("face_registration.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/upload_photo", methods=['POST'])
def upload_photo():
    class_name = request.args.get('class_name')
    path_new_class = os.path.join(DATASET_PATH, class_name)

    # create directory label if not exist
    if not os.path.exists(path_new_class):
        os.mkdir(path_new_class) 
    

    # save uploaded image
    filename = '%03d.jpg' % (len(os.listdir(path_new_class)) + 1) 
    file = request.files['webcam']
    file.save(os.path.join(path_new_class, filename))

    # resize
    img = cv2.imread(os.path.join(path_new_class, filename))
    img = cv2.resize(img, (250, 250))
    cv2.imwrite(os.path.join(path_new_class, filename), img)

    return '', 200

if __name__ == '__main__':
    socketio.run(app, debug=True)