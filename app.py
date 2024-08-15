from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os, sys
import torch

app = Flask(__name__, template_folder='src/templates')
app.config['ALLOWED_EXTENSIONS'] = {'pth', 'tar'}

def get_filename(path):
    return os.path.basename(path)

app.jinja_env.filters['basename'] = get_filename

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def search_pth(folder):
    pth_files = []
    for path, subfolders, files in os.walk(folder):
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(path, file))
    return pth_files

@app.route('/', methods=['GET'])
def index():  
    return render_template('index.html', paths=search_pth("./"))

@app.route('/file/<path:filepath>')
def file_action(filepath):
    
    checkpoint = torch.load(filepath, map_location='cpu')
    for training in checkpoint['history']['training']:
        training.pop('losses', None)
    history = checkpoint['history']

    return render_template('file.html', filename=os.path.basename(filepath), history=history)

if __name__ == '__main__':
    app.run(debug=True)
