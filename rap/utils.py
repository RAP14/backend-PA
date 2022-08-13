import pymongo
from django.http import JsonResponse

client = pymongo.MongoClient("mongodb://Adam:adam141100@cluster0-shard-00-02.yb0qg.mongodb.net:27017/test?replicaSet=atlas-11fubd-shard-0&ssl=true&authSource=admin")
db = client["TryCoins"]
collection = db['coins']

def get_coin(nama_coin):
  datacoin = []
  cursor = collection.find({'nama_coin': nama_coin})
  for data in cursor:
    datacoin.append(data)
  return(datacoin)

# def dataa():
#   # dataku = []
#   # cursor = collection.find()
#   # for data in cursor:
#   #   dataku.append(data)
#   # return (dataku)
#   hehe = [1, 2, 3]
#   hehehe = {"adam": hehe}
#   return JsonResponse(hehehe)