# Code for Flask web application

import requests
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('model.hdf5')
foodList = ["apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare", "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito", "bruschetta", "caesar salad", "cannoli", "caprese salad", "carrot cake", "ceviche", "cheese plate", "cheesecake", "chicken curry", "chicken quesadilla", "chicken wings", "chocolate cake", "chocolate mousse", "churros", "clam chowder", "club sandwich", "crab cakes", "creme brulee", "croque madame", "cup cakes", "deviled eggs", "donuts", "dumplings", "edamame", "eggs benedict", "escargots", "falafel", "filet mignon", "fish and chips", "foie gras", "french fries", "french onion soup", "french toast", "fried calamari", "fried rice", "frozen yogurt", "garlic bread", "gnocchi", "greek salad", "grilled cheese sandwich", "grilled salmon", "guacamole", "gyoza", "hamburger", "hot and sour soup", "hot dog", "huevos rancheros", "hummus", "ice cream", "lasagna", "lobster bisque", "lobster roll sandwich", "macaroni and cheese", "macarons", "miso soup", "mussels", "nachos", "omelette", "onion rings", "oysters", "pad thai", "paella", "pancakes", "panna cotta", "peking duck", "pho", "pizza", "pork chop", "poutine", "prime rib", "pulled pork sandwich", "ramen", "ravioli", "red velvet cake", "risotto", "samosa", "sashimi", "scallops", "seaweed salad", "shrimp and grits", "spaghetti bolognese", "spaghetti carbonara", "spring rolls", "steak", "strawberry shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna tartare", "waffles"]


def predict(img_path):
    """takes user's uploaded image and uses trained model to predict the type of food it is"""
    
    img_width, img_height = 300, 300
    img = image.load_img(img_path, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    i = np.argmax(model.predict(img))
    prediction = foodList[i]

    return prediction


def get_restaurants(dish):
    search_url = "https://api.yelp.com/v3/businesses/search?"
    loc = 'losangeles'
    key ="6w2QtzFsPBim7Xxc8UKy4qAG45hk3aU3_fjHiVoXGvlY2rzjsW-cj_Q_g-EnXvxnnBXGT1wjLL6lRzeS5ji2qszXwYmuobyCq8qsh017JNCh-3xqeokqIgbW5qlbYnYx"
    headers = {'Authorization': 'Bearer ' + key}

    search_url += 'term=' + dish + '&' + 'location=' + loc

    result = requests.get(search_url, headers=headers)

    if result.status_code == 200:
        data = result.json()

        topResults = []
        for i in range(10):
            restaurant = data['businesses'][i]
            topResults += [[restaurant['name'], restaurant['url']]]

    return topResults
