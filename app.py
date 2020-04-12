from flask import Flask 
from flask import request 
from flask import render_template
import predict
import os

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



if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	app.run(debug=True, port=port)