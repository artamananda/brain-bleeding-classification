from flask import Flask, render_template, request
from PIL import Image
from utils import plot_image, run_detection, delete_file, checking_file_format
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    delete_file(app.config['UPLOAD_FOLDER'])
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    delete_file(app.config['UPLOAD_FOLDER'])
    if 'image' not in request.files:
        return render_template('index.html', alert="No file part")
    
    image = request.files['image']
    if image.filename == '':
        return render_template('index.html', alert="No selected file")

    if image and checking_file_format(image.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'brain-img.jpg')
        image.save(filename)
        img = Image.open(filename)
        img_gray = img.convert('L')
        gray_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'brain-img.jpg')
        img_gray.save(gray_filename)
        plot_img = plot_image(filename)
        return render_template('index.html', plot_img=plot_img, alert="File uploaded successfully")
    
    return render_template('index.html', alert="Invalid file format. Allowed formats: jpg, jpeg, png")
    
@app.route('/process', methods=['POST'])
def process():
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'brain-img.jpg')
    if os.path.exists(filename):
        plot_img = plot_image(filename)
        plot_url, detectionStatus = run_detection()
        if detectionStatus:
            return render_template('index.html', plot_img=plot_img, plot_url=plot_url, alert="Detection of brain bleeding has been identified in the image.")
        else:
            return render_template('index.html', plot_img=plot_img, plot_url=plot_url, alert="Healthy Brain Image.")
        
    return render_template('index.html', alert="No selected file")
