import json
import requests
from ip2geotools.databases.noncommercial import DbIpCity

# ip_add = input("Enter IP: ")  
ip_add = "47.9.101.77"
print(ip_add)

data = DbIpCity.get(ip_add, api_key="free")
latitude = data.latitude
longitude = data.longitude

print(latitude)
print(longitude)

weather_data = json.loads(requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid=a6962678a5cba51e8db12b46bc87a867").text)

print(f"temparature = {weather_data['main']['temp'] - 273.15}")
print(f"humadity = {weather_data['main']['humidity']}")