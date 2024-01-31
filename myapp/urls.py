from django.urls import path
from . import views

urlpatterns = [
  path('', views.home, name="home"),
  path('api/process', views.process_file, name='process_file'),
  path('api/check', views.check_file, name='check_file'),
  path('api/extract', views.extract_text, name='extract_text'),
  path('api/keywords', views.extract_keywords, name='extract_keywords'),
]