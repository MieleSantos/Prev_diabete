import numpy as np
import joblib
from flask import Flask, render_template, request,jsonify


app = Flask(__name__)


def previsao_diabetes(lista_valores_formulario):

    #transforma os valores do formulario
    prever = np.array(lista_valores_formulario).reshape(1,8) 
    #realizar a carga do modelo salvo
    modelo_salvo = joblib.load("./models/melhor_modelo.sav") 

    resultado = modelo_salvo.predict(prever)
    
    return resultado[0]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():

    if request.method == "POST":
        # pegando dados do formulario no formato dicionario
        lista_formulario = request.form.to_dict()

       #transformando os dados coletados em uma lista
        lista_formulario = list(lista_formulario.values())
        # convertendo os valores para float
        lista_formulario = list(map(float, lista_formulario))
        #chamando a função para fazer a previsão, passando os dados coletados
        resultado = previsao_diabetes(lista_formulario)

        #resultado da previsão
        if int(resultado)==1:
            previsao = "Possui diabetes"
        else:
            previsao = "Não possui diabetes"

        return render_template("resultado.html", previsao=previsao)

if __name__ == "__main__":
    app.run(debug=True)