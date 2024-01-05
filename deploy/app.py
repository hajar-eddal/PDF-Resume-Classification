# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from classifier.resume_classifier import load_models, classify_resume
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models outside the request context to avoid reloading them on every request
tfidf_vectorizer, kmeans_model, cluster_labels = load_models()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Resume classification
        cluster_label = classify_resume(file_path, tfidf_vectorizer, kmeans_model, cluster_labels)

        return render_template('index.html', cluster_label=cluster_label)

    else:
        flash('Invalid file format. Please upload a PDF file.')
        return redirect(request.url)


if __name__ == '__main__':
    # Use port provided by Heroku or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
