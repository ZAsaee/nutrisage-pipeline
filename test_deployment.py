import requests
import json

url = "http://nutrisage-alb-1160583664.us-east-1.elb.amazonaws.com/api/predict"
payload = {
  "energy_kcal_100g": 350,
  "fat_100g": 15,
  "saturated_fat_100g": 5,
  "carbohydrates_100g": 40,
  "sugars_100g": 20,
  "proteins_100g": 10,
  "sodium_100g": 0.5,
  "fiber_100g": 3,
  "fruit_vegetable_nut_content": 10
}
headers = {
  'Content-Type': 'application/json'
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print(response.json()) 