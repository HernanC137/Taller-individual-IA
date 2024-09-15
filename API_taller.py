import streamlit as st
import pandas as pd
import pickle
from pycaret.regression import predict_model,load_model


#with open('ridge_model.pkl', 'rb') as model_file: #no me sirvio utilizando esta función entonces usa la de pycaret 
#    modelo = pickle.load(model_file)

modelo = load_model('ridge_model')

st.title("Taller Ridge")



Email = st.text_input("Email", value="hcortes356@gmail.com")
address = st.selectbox("Address", options=['Munich', 'Ausburgo', 'Berlin', 'Frankfurt'], index=0)  
dominio = st.selectbox("Dominio", options=['yahoo', 'Otro', 'gmail', 'hotmail'], index=2)  
tec = st.selectbox("Tecnología", options=['PC', 'Smartphone', 'Iphone', 'Portatil'], index=1)
Avg_Session_Length = st.text_input("Avg. Session Length", value="32.063775")
Time_on_App = st.text_input("Time on App", value="10.719")
Time_on_Website = st.text_input("Time on Website", value="37.712")
Length_of_Membership = st.text_input("Length of Membership", value="3.004743")


if st.button("Calcular"):
    try:
        Avg_Session_Length = float(Avg_Session_Length)
        Time_on_App = float(Time_on_App)
        Time_on_Website = float(Time_on_Website)
        Length_of_Membership = float(Length_of_Membership)

        # Crear el dataframe a partir de los inputs del usuario
        user = pd.DataFrame({
            'Email': [Email], 
            'Address': [address], 
            'dominio': [dominio], 
            'Tec': [tec], 
            'Avg. Session Length': [Avg_Session_Length], 
            'Time on App': [Time_on_App], 
            'Time on Website': [Time_on_Website], 
            'Length of Membership': [Length_of_Membership],  
            'price': [0]  
        })


        
        # Asegúrate de usar solo las características que el modelo espera
        user_features = user.drop(columns=['price', 'Email'])  # Excluir columnas no necesarias para la predicción
        
        # Realizar predicciones utilizando el modelo cargado
        predictions = predict_model(modelo, data=user_features)
        
        # Mostrar predicciones
        st.write(f'Predicción de precio: {predictions["prediction_label"][0]}')
    except ValueError:
        st.error("Por favor, ingrese valores numéricos válidos en los campos correspondientes.")

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    st.experimental_rerun()



