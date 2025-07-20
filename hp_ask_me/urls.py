from django.urls import path
from .views import AskQueryView

urlpatterns = [
    path("",AskQueryView.as_view(), name="ask_query")
]