# Food-Identification-using-CNN-

## Overview
This project is a food identification system built using a Convolutional Neural Network (CNN) with TensorFlow and Keras. The model classifies food images into predefined categories and provides nutritional information based on the identified food item. The application is implemented as a web service using Flask.

## Features
- Identifies 20 different food items from images.
- Uses a pre-trained InceptionV3 model fine-tuned for food classification.
- Provides nutritional information including calories, protein, carbs, fats, and fiber.
- Web-based interface for uploading food images and viewing predictions.

## Technologies Used
- **Python**: Programming language for model training and web development.
- **TensorFlow/Keras**: Deep learning framework for model training and inference.
- **Flask**: Web framework for serving the model via an API.
- **NumPy**: Data manipulation and preprocessing.
- **HTML/CSS**: Frontend for user interaction.

## Installation
### Prerequisites
Ensure you have Python installed (preferably 3.7 or later). Install required dependencies using:
```sh
pip install tensorflow flask numpy
```

### Running the Application
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repository/food-identification-cnn.git
   cd food-identification-cnn
   ```
2. Place the trained model file (`model_v1_inceptionV3.h5`) in the project directory.
3. Run the Flask application:
   ```sh
   python app.py
   ```
4. Open a web browser and navigate to `http://127.0.0.1:5000/` to use the application.

## File Structure
```
food-identification-cnn/
│── static/
│   ├── uploads/            # Stores uploaded images
│── templates/
│   ├── index.html          # Homepage template
│   ├── result.html         # Prediction results page
│── app.py                  # Main Flask application
│── model_v1_inceptionV3.h5 # Pre-trained CNN model
│── requirements.txt        # Dependencies
```

## Usage
1. Open the web application.
2. Upload a food image.
3. View the predicted food name and its nutritional details.

## Model Details
- The CNN model is based on **InceptionV3**, fine-tuned for food classification.
- Input image size: **299x299 pixels**.
- The model outputs a probability distribution over 20 food categories.

## Future Improvements
- Expand the dataset to include more food categories.
- Improve accuracy with additional fine-tuning.
- Deploy as a cloud-based API.

## License
This project is open-source and available under the [MIT License](LICENSE).

