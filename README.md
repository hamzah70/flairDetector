# flairDetector

A Flask based Web Application to predict Reddit Flairs
The flair predictor app is live at https://reddit-india-flair-predictor.herokuapp.com/

### Development Environment
To reproduce the code locally run the following command in the particular order
 1. Open the `Terminal`.
 2. Clone the repository using `git clone https://github.com/hamzah70/flairDetector.git`
 3. Open a Terminal
 4. `cd` into the cloned repo
 5. Start a python virtual environment using `pipenv shell` (for windows) and `python3 -m pipenv shell` (for macOS)
 6. Install all the required python dependencies using `pip install -r requirements.txt` (for windows) and `python3 -m pip install -r requirements.txt` (for macOS)
 7. Install nltk data using `python3 -m nltk.downloader stopwords`
 8. Now run the Web Application locally on ypour system by running app.py file using `python3 app.py`

### Automated testing 
To do automated testing for web app use the following command
`curl -i -X POST -F name=Test -F file=@/path/to/txt/file https://reddit-flair-app-predictor.herokuapp.com/automated_testing`
The above command will send a POST request to the endpoint `/automated_testing` with the txt file you provide in place of its path containing links of various reddit posts on different lines and return a json object with postlinks as keys and their predicted flairs as their values.

### Machine Learning model accuracy to predict flairs
0.6106194690265486



