from django.shortcuts import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from joblib import load

# Create your views here.
def index(request):
    return HttpResponse(json.dumps({"message" : "Backend is working"}))

# demo url = http://127.0.0.1:8000/predict/?N=0&P=55&K=22&temperature=22.986669&humidity=20.579406&rainfall=143.858494

@csrf_exempt
def predict(request):
    # digit to crop dict 
    digit_to_crop = {
        1 : 'rice',
        2 : 'maize',
        3 : 'chickpea',
        4 : 'kidneybeans',
        5 : 'pigeonpeas',
        6 : 'mothbeans',
        7 : 'mungbean',
        8 : 'blackgram',
        9 : 'lentil',
        10 : 'pomegranate',
        11 : 'banana',
        12 : 'mango',
        13 : 'grapes',
        14 : 'watermelon',
        15 : 'muskmelon',
        16 : 'apple',
        17 : 'orange',
        18 : 'papaya',
        19 : 'coconut',
        20 : 'cotton',
        21 : 'jute',
        22 : 'coffee'
    }
    # print(os.listdir())
    # Load and predict crop
    input_values = [request.GET.get('N'),request.GET.get('P'),request.GET.get('K'),request.GET.get('temperature'),request.GET.get('humidity'),request.GET.get('rainfall')]
    # print(input_values)
    loaded_model = load("home/CRS_RandomForest_model.joblib")
    output = loaded_model.predict([input_values])
    return HttpResponse(json.dumps({"message" : "Backend is working",
                                    "crop" : f"{digit_to_crop[int(output)]}"}))