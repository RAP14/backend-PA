from django.urls import path, include
from django.http import JsonResponse
from . import views

urlpatterns = [
    path('coin/<str:id>', views.get_coin, name='get_coin'),
    path('data/<str:id>', views.get_data, name='get_data'),
    path('lstm/<str:id>/<int:day>', views.get_prediction, name='get_prediction'),
    
    # path('home/', views.home, name='home'),
    # path('data', views.get_all_coin, name='get_all_coin'),
]
