import os
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your model
model = load_model("J:\Mini project\Food identification\model_v1_inceptionV3.h5")

# Define your category mapping
category = {
    0: ['burger', 'Burger'],
    1: ['butter_naan', 'Butter Naan'],
    2: ['chai', 'Chai'],
    3: ['chapati', 'Chapati'],
    4: ['chole_bhature', 'Chole Bhature'],
    5: ['dal_makhani', 'Dal Makhani'],
    6: ['dhokla', 'Dhokla'],
    7: ['fried_rice', 'Fried Rice'],
    8:['idli', 'Idli'],
    9: ['jalebi', 'Jalebi'],
    10: ['kathi_rolls', 'Kathi Rolls'],
    11: ['kadai_paneer', 'Kadai Paneer'],
    12: ['kulfi', 'Kulfi'],
    13: ['masala_dosa', 'Masala Dosa'],
    14: ['momos', 'Momos'],
    15: ['paani_puri', 'Paani Puri'],
    16: ['pakode', 'Pakode'],
    17: ['pav_bhaji', 'Pav Bhaji'],
    18: ['pizza', 'Pizza'],
    19: ['samosa', 'Samosa']
}

# Nutrition information for food items
food_nutrition = {
    'burger': {'calories': 300, 'protein': 15, 'carbs': 30, 'fats': 18, 'fiber': 2},
    'butter_naan': {'calories': 250, 'protein': 7, 'carbs': 40, 'fats': 6, 'fiber': 1},
    'chai': {'calories': 100, 'protein': 2, 'carbs': 20, 'fats': 2, 'fiber': 0},
    'chapati': {'calories': 340, 'protein': 10, 'carbs': 60, 'fats': 5, 'fiber': 3},
    'chole_bhature': {'calories': 500, 'protein': 15, 'carbs': 80, 'fats': 20, 'fiber': 7},
    'dal_makhani': {'calories': 450, 'protein': 20, 'carbs': 50, 'fats': 20, 'fiber': 5},
    'dhokla': {'calories': 150, 'protein': 5, 'carbs': 25, 'fats': 3, 'fiber': 1},
    'fried_rice': {'calories': 350, 'protein': 8, 'carbs': 60, 'fats': 10, 'fiber': 2},
    'idli': {'calories': 150, 'protein': 5, 'carbs': 30, 'fats': 2, 'fiber': 1},
    'jalebi': {'calories': 300, 'protein': 4, 'carbs': 70, 'fats': 10, 'fiber': 0},
    'kathi_rolls': {'calories': 400, 'protein': 15, 'carbs': 50, 'fats': 15, 'fiber': 4},
    'kadai_paneer': {'calories': 450, 'protein': 20, 'carbs': 40, 'fats': 30, 'fiber': 3},
    'kulfi': {'calories': 200, 'protein': 6, 'carbs': 20, 'fats': 10, 'fiber': 0},
    'masala_dosa': {'calories': 300, 'protein': 8, 'carbs': 50, 'fats': 10, 'fiber': 3},
    'momos': {'calories': 250, 'protein': 10, 'carbs': 35, 'fats': 5, 'fiber': 2},
    'paani_puri': {'calories': 150, 'protein': 2, 'carbs': 30, 'fats': 5, 'fiber': 1},
    'pakode': {'calories': 200, 'protein': 4, 'carbs': 20, 'fats': 12, 'fiber': 2},
    'pav_bhaji': {'calories': 350, 'protein': 10, 'carbs': 60, 'fats': 15, 'fiber': 5},
    'pizza': {'calories': 400, 'protein': 15, 'carbs': 50, 'fats': 18, 'fiber': 3},
    'samosa': {'calories': 250, 'protein': 6, 'carbs': 30, 'fats': 12, 'fiber': 2},
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', prediction="No file uploaded", calories=None, nutrition=None, image_file=None)

    file = request.files['file']

    if file.filename == '':
        return render_template('result.html', prediction="No file uploaded", calories=None, nutrition=None, image_file=None)

    # Save the uploaded file
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Make a prediction
    food_name, nutrition_info = predict_image_with_nutrition(file_path)

    return render_template('result.html', prediction=food_name, calories=nutrition_info['calories'], nutrition=nutrition_info, image_file=file.filename)

def predict_image_with_nutrition(filename):
    img_ = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    
    food_name = category.get(index)[0]  # Get food name based on index
    nutrition_info = food_nutrition.get(food_name, None)  # Get nutrition info

    return food_name, nutrition_info

if __name__ == '__main__':
    app.run(debug=True)
