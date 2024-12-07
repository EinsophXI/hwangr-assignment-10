from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from search import perform_image_search, perform_text_search, perform_hybrid_search

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('search_results', filename=filename))

    return "File not allowed", 400


@app.route('/search', methods=['POST'])
def search():
    query_type = request.form.get('query_type')
    query_text = request.form.get('query_text', '')
    uploaded_image = request.files.get('image')
    lam = request.form.get('lam', type=float)  # Get the lam value for hybrid search
    embedding_type = request.form.get('embedding_type', 'clip')  # Default to 'clip'

    use_pca = embedding_type == 'pca'

    if query_type == 'text':
        results = perform_text_search(query_text, use_pca=use_pca)
    elif query_type == 'image' and uploaded_image:
        filename = secure_filename(uploaded_image.filename)
        uploaded_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        results = perform_image_search(os.path.join(app.config['UPLOAD_FOLDER'], filename), use_pca=use_pca)
    elif query_type == 'hybrid' and uploaded_image:
        filename = secure_filename(uploaded_image.filename)
        uploaded_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        results = perform_hybrid_search(query_text, os.path.join(app.config['UPLOAD_FOLDER'], filename), lam, use_pca=use_pca)
    else:
        results = []  # No valid search type or missing inputs

    return render_template('index.html', results=results)



@app.route('/coco_images_resized/<filename>')
def serve_image(filename):
    image_folder = os.path.join(app.root_path, 'coco_images_resized')  # Custom folder
    return send_from_directory(image_folder, filename)


if __name__ == '__main__':
    app.run(debug=True)
