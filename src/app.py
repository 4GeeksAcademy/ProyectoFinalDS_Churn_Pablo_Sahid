from utils import db_connect
engine = db_connect()

# your code here

import streamlit as st
import joblib
import pandas as pd 
import json 

model = joblib.load("/workspaces/ProyectoFinalDS_Churn_Pablo_Sahid/models/best_model_XGBoost.joblib")

#titulo de la pagina 
st.title("Predictor de Churn")

#el usuario sube un fichero csv

csv_usuario =  st.file_uploader(label = "Suba un archivo csv para realizar la predicción",
                    type = "csv",
                    help = "Solo se admiten archivos csv")

if csv_usuario is not None:                          #si tenemos el csv subido
    dataframe = pd.read_csv(csv_usuario)             #creamos un dataframe a partir de el
    clientes = dataframe["CustomerID"].copy()        #se va a utilizar al final del proceso para mostrar el id de los clientes en la prediccion
    st.write("Archivo CSV subido correctamente.")    #notificamos al usuario 

                                                     #se comprueba que el csv del usuario tenga nulos
    columnas_con_nulos = dataframe.columns[dataframe.isnull().any()].tolist()
    if columnas_con_nulos:
        st.error("El CSV contiene valores nulos. Por favor, sube un CSV sin valores nulos.") # si hay nulos lanzamos mensaje de error
        st.stop()                                                                            # y paramos la ejecucion de la pagina 

    if st.button("Predecir"):                        #creamos boton para iniciar la prediccion 
        st.write("estoy prediciendo!")               #notificamos que se esta prediciendo

        #lista de variables predictoras 
        data_a_predecir = ['MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','DirectorAssistedCalls','OverageMinutes',
                           'RoamingCalls','PercChangeMinutes','PercChangeRevenues','DroppedCalls','BlockedCalls',
                            'UnansweredCalls','CustomerCareCalls','ThreewayCalls','ReceivedCalls','OutboundCalls',
                            'InboundCalls','PeakCallsInOut','OffPeakCallsInOut','CallForwardingCalls','CallWaitingCalls',
                            'MonthsInService','UniqueSubs','ActiveSubs','Handsets','HandsetModels','CurrentEquipmentDays',
                            'AgeHH1','AgeHH2','RetentionCalls','RetentionOffersAccepted','ReferralsMadeBySubscriber',
                            'IncomeGroup','AdjustmentsToCreditRating','ServiceArea_n','ChildrenInHH_n','HandsetRefurbished_n',
                            'HandsetWebCapable_n','TruckOwner_n','RVOwner_n','Homeownership_n','BuysViaMailOrder_n',
                            'RespondsToMailOffers_n','OptOutMailings_n','NonUSTravel_n','OwnsComputer_n','HasCreditCard_n',
                            'NewCellphoneUser_n','OwnsMotorcycle_n','HandsetPrice_n','MadeCallToRetentionTeam_n',
                            'CreditRating_n','PrizmCode_n','Occupation_n','MaritalStatus_n']
       
        # Eliminar las 3 columnas sobrantes de IDs o con información irrelevante
        dataframe = dataframe.drop(columns = ['CustomerID','DroppedBlockedCalls','NotNewCellphoneUser'])

        # Quedarse solo con las 3 primeras letras de la variable ServiceArea
        dataframe['ServiceArea'] = dataframe['ServiceArea'].str[:3]

        # --------------------------factorizacion del dataframe del usuario-----------------------------------------
         
        #funcion para encontrar el diccionario con las reglas de conversion 
        def get_diccionario(columna):
            #encontrar el json correspondiente
            json_string = f"/workspaces/ProyectoFinalDS_Churn_Pablo_Sahid/models/{columna}_transformation_rules.json"
            with open(json_string) as f:
              diccionario = json.load(f)
            return diccionario

        #una vez tenemos el diccionario con las reglas de factorizacion, la aplicamos en la columna correspondiente del dataframe
        #lista de las variables categoricas que necesitan ser factorizadas
        lista_categoricas = [
        'ServiceArea', 'ChildrenInHH', 'HandsetRefurbished',
        'HandsetWebCapable', 'TruckOwner', 'RVOwner', 'Homeownership',
        'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings',
        'NonUSTravel', 'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser',
        'OwnsMotorcycle', 'MadeCallToRetentionTeam',
        'MaritalStatus', 'PrizmCode', 'Occupation', 'CreditRating',
        'HandsetPrice'
        ]
        #recorremos las columnas categoricas del dataframe
        for col in lista_categoricas:
            #buscar diccionario con las reglas de conversion
            reglas_conversion = get_diccionario(col)
            #aplicar la factorizacion en la columna correspondiente del dataframe
            #en el df original, conservamos las columnas tipo string y creamos nuevas con el sufijo _n, asi que hacemos lo mismo aqui 
            dataframe[col + "_n"] = dataframe[col].map(reglas_conversion)

        #prediccion
        prediction = model.predict(dataframe[data_a_predecir])

        # se imprime el resultado
        #st.write("Prediction" , prediction) comentado codigo o manera de mostrar resultados anterior.

        # Probabilidad de cada clase
        probabilidades = model.predict_proba(dataframe[data_a_predecir])

        # Probabilidad de Churn (clase 0)
        probabilidad_churn = probabilidades[:, 0]

        # Crear dataframe de resultados
        resultados = pd.DataFrame({"Cliente": clientes,"Churn": prediction,"Probabilidad de Churn": probabilidad_churn})

        # Convertir 0/1 a Si/No
        resultados["Churn"] = resultados["Churn"].map({0: "Sí",1: "No"})

        # Convertir probabilidad a porcentaje
        resultados["Probabilidad de Churn"] = resultados["Probabilidad de Churn"].map(lambda x: f"{x:.1%}")

        # Mostrar resultados
        st.subheader("Resultados de la predicción")

        st.dataframe(resultados,use_container_width=True)

        
        # Descargar resultados

        csv_resultados = resultados.to_csv(index=False).encode("utf-8")

        st.download_button(label="📥 Descargar resultados en CSV",data=csv_resultados,file_name="predicciones_churn.csv",mime="text/csv")
