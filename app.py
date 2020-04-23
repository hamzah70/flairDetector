from flask import Flask 
from flask import request 
from flask import render_template
import predict
import os
from werkzeug import secure_filename
import json

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('webapp.html', flair = "Not yet predicted")




@app.route('/', methods=['POST'])
def my_form_post():
	url = request.form['url']
	flairPredicted = predict.recoverPost(url)
	print(flairPredicted)
	return render_template('webapp.html', flair = flairPredicted)

@app.route('/automated_testing', methods=['POST'])
def testing():
	txtFile = request.files['upload_file']
	allurl = txtFile.read().decode('utf-8').split('\n')
	results = {}
	for url in allurl:
		flairPredicted = predict.recoverPost(url)
		results[url] = flairPredicted
	return json.dumps(results)
	



if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	app.run(debug=True, port=port)