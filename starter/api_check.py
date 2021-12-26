import requests

url = 'https://udacity-c3-fastapi.herokuapp.com/'
obj = {
  "age": '39',
  "workclass": "State-gov",
  "fnlgt": '77516',
  "education": "Bachelors",
  "education_num": '13',
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": '2174',
  "capital_loss": '0',
  "hours_per_week": '0',
  "native_country": "United-States"
}

x = requests.post(url, json=obj)

print(x.status_code)
print(x.text)
