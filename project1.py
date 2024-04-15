import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from io import StringIO
from pdfminer.high_level import extract_text
import pdfminer
import io
import tkinter as tk
from tkinter import filedialog

#load jobs3.csv data
job_data = pd.read_csv("test1.csv")
job_data.fillna('missing', inplace=True)
def load_pdf():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    #extract_text from resume_file,we can check using print(resume_text) 
    with open(file_path, 'rb') as resume_file:
        resume_text = extract_text(resume_file)
    #vectorizer.transform gives a resumne matrix containing token count of each resume_text
    #we can check this by print(resume_matrix)
    resume_matrix = vectorizer.transform([resume_text])
    print("resume_matrix")
    print(pd.DataFrame(resume_matrix.toarray(), columns=vectorizer.get_feature_names_out()))
    #predict_proba is a method in navie bayes classifier which is trained with job_requirements to predict probability of each token
    probabilities = classifier.predict_proba(resume_matrix)
    n = 3
    top_n_predictions = sorted(zip(classifier.classes_, probabilities[0]), key=lambda x: x[1], reverse=True)[:n]
    print(top_n_predictions) 
    for prediction in (top_n_predictions):
        job_label = tk.Label(root, text=prediction[0], font=("white", 14), fg="black", padx=10, pady=2)
        job_label.pack()
        
vectorizer = CountVectorizer()
job_matrix = vectorizer.fit_transform(job_data["Requirements"])
classifier = MultinomialNB()
#training classifier
classifier.fit(job_matrix, job_data["Job"])
root = tk.Tk()
root.geometry("800x800")
root.title("Job Recommendation System")
srecommendations_label = tk.Label(root, text="Job Recommendation system", font=("Arial", 50), bg="#f2f2f2")
recommendations_label.pack(pady=10)
txt_label = tk.Label(root, text="  We provide you Job Recommendations based on your Resume", font=("Arial", 30), bg="black",fg="white")
txt_label.pack(padx=200,pady=8)
browse_button = tk.Button(root, text="Browse Resume", height=2,width=30,font=("Arial", 12), bg="#3c8dbc", fg="#fff", command=load_pdf)
browse_button.pack(pady=210)
txt1_label = tk.Label(root, text="Job Recommendations :  ", font=("Arial", 20), bg="black",fg="white")
txt1_label.pack(padx=200,pady=3)
root.configure(bg='skyblue')
root.mainloop()







