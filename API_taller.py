import streamlit as st
import pandas as pd
import pickle
from pycaret.regression import predict_model,load_model


with open('ridge_model.pkl', 'rb') as model_file: 
    modelo = pickle.load(model_file)


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
            'Avg. Session Length': [Avg_Session_Length], 
            'Time on App': [Time_on_App], 
            'Time on Website': [Time_on_Website], 
            'Length of Membership': [Length_of_Membership],
            'Email': [Email], 
            'Address': [address], 
            'dominio': [dominio], 
            'Tec': [tec], 
            'price': [0]  
        })


        
        #  quita las variables que no sirven
        data_pred = user.drop(columns=['price', 'Email'])
        
        
        predictions = predict_model(modelo, data=data_pred)
        
        # Mostrar predicciones
        st.write(f'Predicción de precio: {predictions["prediction_label"][0]}')
        
    except ValueError:
        st.error("Por favor, ingrese valores numéricos válidos en los campos correspondientes.")

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    st.experimental_rerun()



