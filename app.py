from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib

import json

# load the built-in model 
#gbr = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "HI"

@app.route('/api') #, methods=['POST'])
def get_delay():
	#result=request.form
	#height = result['height']
	#width = result['width']
	#shoe_size = result['shoe_size']
	#fiscal_power = result['fiscal_power']
	#fuel_type = result['fuel_type']
	# we create a json object that will hold data from user inputs
	#user_input = {'height':height, 'width':width, 'shoe_size':shoe_size}
	# encode the json object to one hot encoding so that it could fit our model
	X_new = [[154, 54, 37]]
	#print('New Sample:', X_new)
	#print('Predicted class:', gbr.predict(X_new))
	#a = input_to_one_hot(user_input)
	# get the price prediction
	#price_pred = gbr.predict([a])[0]
	#price_pred = round(price_pred, 2)
	# return a json value
	# return json.dumps({'gender':price_pred});
	gbr = joblib.load('model.pkl')
	return gbr.predict(X_new);

if __name__ == '__main__':
    app.run(port=8177, debug=True)