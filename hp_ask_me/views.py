from django.shortcuts import render
from django.views import View
from rag_system.rag_retrive import retrive_answer
from rag_system.rag_db_create import create_vec_db
import os

class AskQueryView(View):
    def get(self, request):
        return render(request, "home.html")

    def post(self, request):
        question = request.POST.get("query")
        string = create_vec_db()
        response = retrive_answer(question)
        context = {"response": response, "question": question}
        return render(request, "home.html", context)
