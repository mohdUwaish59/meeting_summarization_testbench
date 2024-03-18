# meeting_summarization_portal/api/urls.py

from django.urls import path
from .views import SummarizationAPIView #,VisualizationAPIView
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('summarize/', SummarizationAPIView.as_view(), name='summarize'),
    #path('visualization/', views.VisualizationAPIView.as_view(), name='visualization'),
    #path('report/', VisualizationAPIView.as_view(), name='report'),
    
]
