import requests

url = 'http://192.168.1.109:80'
img =  open('test_data\\images\\test.jpg','rb')
my_image = {'image': img}
r = requests.post(url, files=my_image)

# convert server response into JSON format.
print(r.json())