from flask import Flask, render_template, request, redirect, send_from_directory, url_for, render_template_string
import numpy as np
import json
import uuid
import tensorflow as tf
import os

from contour import find_leaf_contours  # <-- Add this line

app = Flask(__name__)
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
label = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Single diseased plant leaf allowed',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image.save(f'{temp_name}_{image.filename}')  # Save original
        img_path = f'{temp_name}_{image.filename}'
        prediction = model_predict(f'./{img_path}')
        contour_img_path = 'contour_output.jpg'  # Always save with this name
        find_leaf_contours(img_path, contour_img_path)
        return render_template(
            'home.html',
            result=True,
            imagepath=f'/{img_path}',
            prediction=prediction,
            contourimg='/contour_output.jpg'  # Always reference this file
        )
    else:
        return redirect('/')

@app.route('/<path:filename>')
def serve_any(filename):
    return send_from_directory('.', filename)

# --- Feature card routes: render_template_string (only one definition per route!) ---
@app.route('/fast-disease-detection')
def fast_disease_detection():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Fast Disease Detection</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css">
      <style>
      body {
        background: linear-gradient(99deg, #20e295 0%, #2e8cff 120%);
        font-family: 'Segoe UI', Arial, sans-serif;
        margin:0; padding:0;
      }
      .fdd-card-modern {
        max-width: 470px;
        margin: 9vh auto 0 auto;
        background: #fff;
        border-radius: 20px;
        box-shadow: 0 8px 36px #179c6b16;
        border: 1.4px solid #20e29533;
        padding: 2.8em 2em 2.3em 2em;
        text-align: center;
        color: #15836a;
      }
      .fdd-card-modern h2 {
        color: #1789e2;
        margin-bottom: 0.65em;
        font-size: 2em;
        font-weight: 700;
        letter-spacing: 1.2px;
      }
      .fdd-desc-modern {
        font-size: 1.15em;
        color: #138966;
        margin-bottom: 1.4em;
        font-weight: 500;
      }
      .fdd-list-modern {
        text-align: left;
        color: #186c55;
        margin: 0.7em 0 1.6em 0;
        font-size: 1.08em;
      }
      .fdd-list-modern li {
        margin-bottom: 0.6em;
      }
      .ai-badge {
        display: inline-block;
        background: linear-gradient(90deg,#13ab74,#2e8cff 80%);
        color: #fff;
        border-radius: 12px;
        padding: 0.34em 1.2em;
        font-weight: 600;
        font-size: 1em;
        letters-spacing: 0.5px;
        margin-bottom: 1.1em;
      }
      .back-link-modern {
        margin-top: 2.2em;
        display:inline-block;
        font-size:1.04em;
        font-weight:500;
        color:#1789e2;
        text-decoration:underline;
      }
      </style>
    </head>
    <body>
      <div class="fdd-card-modern">
        <h2>Fast Disease Detection</h2>
        <div class="ai-badge"><i class="fa-solid fa-microchip"></i> AI Powered</div>
        <div class="fdd-desc-modern">
            Instantly analyze leaf images for signs of plant disease using advanced Computer Vision and Machine Learning models.
        </div>
        <ul class="fdd-list-modern">
          <li>Upload a clear image of your plant or leaf.</li>
          <li>AI will process and identify possible diseases in seconds.</li>
          <li>You receive instant results & treatment advice, all online.</li>
          <li>All predictions are based on thousands of expertly-labeled training images.</li>
        </ul>
        <a href="/" class="back-link-modern">&larr; Back to Home</a>
      </div>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/dashboard-history')
def dashboard_history():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Dashboard & History</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css">
      <style>
      body {
        background: linear-gradient(118deg,#e9ebff 0%,#d1fff4 100%);
        font-family: 'Segoe UI', Arial, sans-serif;
        margin:0;padding:0;
      }
      .history-card-modern {
        max-width: 490px;
        margin: 8vh auto 0 auto;
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 8px 32px #2943961a;
        border: 1.7px solid #b5dafc;
        padding: 2.9em 2.2em 2.4em 2.2em;
        text-align: center;
        color: #234a7a;
      }
      .history-card-modern h2 {
        color: #237a6b;
        margin-bottom: 0.6em;
        font-size: 2em;
        font-weight: 700;
      }
      .history-desc-modern {
        font-size: 1.15em;
        color: #234a7a;
        margin-bottom: 1.2em;
        font-weight: 500;
      }
      .dev-credit-modern {
        background: linear-gradient(90deg,#2e8cff33,#8eedff35);
        border-radius: 13px;
        margin: 2em 0 0.7em 0;
        padding: 1.05em 1.2em;
        color: #164faf;
        font-size: 1.03em;
        font-weight: 500;
        letter-spacing: 0.5px;
      }
      .dev-credit-modern span {
        color: #d53453;
        font-weight: bold;
      }
      .back-link-modern {
        margin-top: 1.6em;
        display:inline-block;
        font-size:1.04em;
        font-weight:500;
        color:#237a6b;
        text-decoration:underline;
      }
      .small-note {
        color: #7d7f98;
        font-size: 0.98em;
        margin-top: 1em;
      }
      </style>
    </head>
    <body>
      <div class="history-card-modern">
        <h2>Dashboard & History</h2>
        <div class="history-desc-modern">
          Track your past diagnoses, monitor plant health trends, and review your results and recommendations in one place.
        </div>
        <div style="margin-bottom:1.3em;">
          <i class="fa-solid fa-chart-line fa-3x" style="color:#13ab74;"></i>
        </div>
        <div class="dev-credit-modern">
          Developed by <span>T.Omkar Saicharan</span> &mdash; M.Tech CSE-AIDS, KL University<br>
          Powered by <b>Machine Learning</b> and advanced training images
        </div>
        <div class="small-note">
            Data is processed securely and only accessible to you.<br>
            Your dashboard will grow as you use our platform!
        </div>
        <a href="/" class="back-link-modern">&larr; Back to Home</a>
      </div>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/treatment-guidance')
def treatment_guidance():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Treatment Guidance</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css">
      <style>
      body {
        background: linear-gradient(120deg,#f7fff0,#e9e6fa 97%);
        font-family: 'Segoe UI', Arial, sans-serif;
        margin:0;padding:0;
      }
      .treatment-card-modern {
        max-width: 460px;
        margin: 9vh auto 0 auto;
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 6px 32px #e18a6bc2;
        border: 1.2px solid #ffd6b5;
        padding: 2.7em 2em 2.1em 2em;
        text-align: center;
        color: #905c2b;
      }
      .treatment-card-modern h2 {
        color: #e18a6b;
        margin-bottom: 0.7em;
        font-size: 2em;
        font-weight: 700;
      }
      .treatment-desc-modern {
        font-size: 1.15em;
        color: #8c6930;
        margin-bottom: 1.2em;
        font-weight: 500;
      }
      .treatment-list-modern {
        text-align: left;
        margin: 1em 0 2.2em 0;
        font-size: 1.06em;
        color: #6d402d;
      }
      .treatment-list-modern li {
        margin-bottom: 0.5em;
      }
      .back-link-modern {
        margin-top: 1.1em;
        display:inline-block;
        font-size:1.04em;
        font-weight:500;
        color:#e18a6b;
        text-decoration:underline;
      }
      </style>
    </head>
    <body>
      <div class="treatment-card-modern">
        <h2>Treatment Guidance</h2>
        <div class="treatment-desc-modern">
            Get actionable advice for treating detected diseases.
        </div>
        <ul class="treatment-list-modern">
          <li><strong>Step 1:</strong> Identify the affected plant and disease type.</li>
          <li><strong>Step 2:</strong> Remove severely affected plant parts (leaves, stems, fruit).</li>
          <li><strong>Step 3:</strong> Apply appropriate fungicides or pesticides as recommended.</li>
          <li><strong>Step 4:</strong> Ensure good sanitation and avoid water splash on leaves.</li>
          <li><strong>Step 5:</strong> Monitor plant regularly and repeat treatment as needed.</li>
          <li><strong>Resources:</strong> <a href="https://www.cabi.org/isc/diseases">CABI Plant Disease Database</a></li>
        </ul>
        <a href="/" class="back-link-modern">&larr; Back to Home</a>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/user-profile-settings')
def user_profile_settings():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>User Profile & Settings</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css">
      <style>
      body {
        background: linear-gradient(120deg,#d1ffee,#fffbe7 95%);
        font-family: 'Segoe UI', Arial, sans-serif;
        margin:0;padding:0;
      }
      .profile-card-modern {
        max-width: 450px;
        margin: 8vh auto 0 auto;
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 6px 32px #179c6b1c;
        border: 1.2px solid #b1e1c6;
        padding: 2.7em 2em 2.1em 2em;
        text-align: center;
        color: #31795b;
      }
      .profile-card-modern h2 {
        color: #179c6b;
        margin-bottom: 0.7em;
        font-size: 2em;
        font-weight: 700;
      }
      .profile-desc-modern {
        font-size: 1.15em;
        color: #0b7c4d;
        margin-bottom: 1.2em;
        font-weight: 500;
      }
      .profile-details-list {
        text-align: left;
        margin: 1em 0;
        font-size: 1.06em;
        color: #256958;
      }
      .profile-details-list li {
        margin-bottom: 0.5em;
        padding-left: 0.4em;
      }
      .edit-btn-modern {
        background: linear-gradient(90deg,#13ab74,#179c6b 70%);
        color: #fff;
        font-weight: 600;
        border: none;
        border-radius: 100px;
        padding: 0.8em 2.5em;
        margin-top: 0.7em;
        margin-bottom: 1em;
        font-size: 1.08em;
        box-shadow: 0 2px 8px #14e29518;
        transition: background 0.22s;
      }
      .edit-btn-modern:hover {
        background: linear-gradient(90deg,#2e8cff,#14e295 90%);
        color: #fff;
      }
      .back-link-modern {
        margin-top: 1.8em;
        display:block;
        font-size:1.04em;
        font-weight:500;
        color:#179c6b;
        text-decoration:underline;
      }
      </style>
    </head>
    <body>
      <div class="profile-card-modern">
        <h2>User Profile & Settings</h2>
        <div class="profile-desc-modern">
            Personalize your experience, manage preferences securely, and update your account details.
        </div>
        <ul class="profile-details-list">
            <li><strong>Name:</strong> Your Name Here</li>
            <li><strong>Email:</strong> youremail@example.com</li>
            <li><strong>Preferences:</strong> Notifications: On, Theme: Light, Language: English</li>
        </ul>
        <button class="edit-btn-modern">Edit Profile (Coming Soon)</button>
        <a href="/" class="back-link-modern">&larr; Back to Home</a>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/support-contact')
def support_contact():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Support & Contact</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css">
      <style>
      body {
        background: linear-gradient(120deg,#d1ffee,#fffbe7 95%);
        font-family: 'Segoe UI', Arial, sans-serif;
        margin:0;padding:0;
      }
      .contact-card-modern {
        max-width: 440px;
        margin: 8vh auto 0 auto;
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 6px 32px #179c6b1c;
        border: 1.2px solid #b1e1c6;
        padding: 2.7em 2em 2.1em 2em;
        text-align: center;
        color: #31795b;
      }
      .contact-card-modern h2 {
        color: #179c6b;
        margin-bottom: 0.7em;
        font-size: 2em;
        font-weight: 700;
      }
      .contact-card-modern .feature-desc {
        font-size: 1.15em;
        color: #0b7c4d;
        margin-bottom: 1.4em;
        font-weight: 500;
      }
      .contact-links-modern a {
        margin: 0.44em;
        padding: 0.83em 2em 0.83em 2em;
        border-radius: 100px;
        background: linear-gradient(90deg,#13ab74,#179c6b 70%);
        color: #fff;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        transition: background 0.22s;
        box-shadow: 0 2px 8px rgba(20,170,120,.09);
      }
      .contact-links-modern a:hover {
        background: linear-gradient(90deg,#2e8cff,#14e295 90%);
        color: #fff;
      }
      .back-link-modern {
        margin-top: 1.8em;
        display:block;
        font-size:1.04em;
        font-weight:500;
        color:#179c6b;
        text-decoration:underline;
      }
      </style>
    </head>
    <body>
      <div class="contact-card-modern">
        <h2>Support & Contact</h2>
        <div class="feature-desc">Chat with plant health experts or reach out directly for help.</div>
        <hr>
        <div class="contact-links-modern">
          <a href="mailto:planthelp@example.com">Email Support</a>
          <a href="tel:+1234567890">Call Hotline</a>
          <a href="#">Live Chat (Coming Soon!)</a>
        </div>
        <a href="/" class="back-link-modern">&larr; Back to Home</a>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)


if __name__ == "__main__":
    app.run(debug=True)
