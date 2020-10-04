from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from model import get_words


app = Flask(__name__)

model = load_model("shake.h5")


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/generate', methods=["POST"])
def generate():
	string = request.form.get('text')
	number = request.form['nextstep']
	if number is '': number = 35
	if string is '': string = 'Oh Romeo, Oh Romeo,'
	prediction = get_words(model, string, int(number))
	result="Result"
	return render_template('index.html', prediction=prediction, result=result)

if __name__ == "__main__":
    app.run(debug=True)

