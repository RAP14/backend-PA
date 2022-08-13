from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from scipy.fft import idctn
from scipy.fftpack import idct
from rap import data, utils, lstm

def get_coin(request, id):
    return HttpResponse(utils.get_coin(id))

def get_data(request, id):
    return HttpResponse(data.get_data(id))

def get_prediction(request, id, day):
    return HttpResponse(lstm.get_prediction(id, day))

# Create your views here.

# def home (request) :
#     return HttpResponse("Hello Rap")

# def get_all_coin(request):
#     return HttpResponse(utils.dataa())
