from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
from model import predict_image
import random

app = Flask(__name__)

# Mapping of the choices
choices = {0: 'rock', 1: 'paper', 2: 'scissors'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Read image file in memory
            img_bytes = file.read()
            img_stream = BytesIO(img_bytes)

            # Call the prediction function from model.py
            user_choice = predict_image(img_stream)

            # Generate a random choice for the computer
            computer_choice_index = random.randint(0, 2)
            computer_choice = choices[computer_choice_index]

            # Determine the winner
            result = determine_winner(user_choice, computer_choice)

            return jsonify({'user_choice': user_choice, 'computer_choice': computer_choice, 'result': result})
    
    except OSError as e:
        return jsonify({'error': 'An error occurred while processing the image. Please try again with a different image or smaller size.'})

def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "It's a tie!"
    elif (user_choice == 'rock' and computer_choice == 'scissors') or \
         (user_choice == 'paper' and computer_choice == 'rock') or \
         (user_choice == 'scissors' and computer_choice == 'paper'):
        return "You win!"
    else:
        return "Computer wins!"

if __name__ == '__main__':
    app.run(debug=True)
