from rag_system.rag_db_create import create_vec_db
from rag_system.rag_retrive import retrive_answer
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    create_vec_db(current_dir)
    query = input("enter your query releted to harry potter")
    response = retrive_answer(query, current_dir)
    print(response)