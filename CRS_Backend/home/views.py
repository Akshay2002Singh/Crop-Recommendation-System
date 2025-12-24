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
    
    try:
        # Get and convert input values to float
        n = float(request.GET.get('N', 0))
        p = float(request.GET.get('P', 0))
        k = float(request.GET.get('K', 0))
        temp = float(request.GET.get('temperature', 0))
        hum = float(request.GET.get('humidity', 0))
        rain = float(request.GET.get('rainfall', 0))
        input_values = [n, p, k, temp, hum, rain]

        # Load model
        loaded_model = load("home/CRS_RandomForest_model.joblib")
        
        # Compatibility hack for different scikit-learn versions
        if hasattr(loaded_model, 'estimators_'):
            for est in loaded_model.estimators_:
                if not hasattr(est, 'monotonic_cst'):
                    est.monotonic_cst = None
        
        # Predict crop
        output = loaded_model.predict([input_values])
        crop_id = int(output[0])
        
        return HttpResponse(json.dumps({
            "message": "Prediction successful",
            "crop": digit_to_crop.get(crop_id, "unknown")
        }), content_type="application/json")
        
    except Exception as e:
        return HttpResponse(json.dumps({
            "error": str(e)
        }), status=500, content_type="application/json")
