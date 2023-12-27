import requests

data = requests.get("http://crs.pythonanywhere.com/predict/?N=0&P=55&K=22&temperature=22.986669&humidity=20.579406&rainfall=143.858494")

print(data.json())