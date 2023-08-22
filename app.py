import streamlit as st
import pandas as pd 
import pickle 

# Useful Functions
def load_ml_components(fp):
    "Load the ml components to re-use in app"
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object

# Variables and Constants
#ml_core_fp = "model.pk" 

# Execution
#ml_components_dict = load_ml_components(fp=ml_core_fp)

#labels = ml_components_dict['labels']
#idx_to_labels = {i: l for (i,l) in enumerate(labels)}

# Function to load the dataset
@st.cache_resource
def load_data(relative_path):
   data= pd.read_csv(relative_path, index_col= 0)
   #merged["date"] = pd.to_datetime(merged["date"])
   return data

    


# Loading the base dataframe
rpath = r"merged_train_data.csv"
data = load_data(rpath)



# Load the model and encoder and scaler
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Interface 
st.write("""
# Sales Forecasting App
Fill form and predict!
As easy as that! 👍🏾 
""")
 
# Inputs 
date = st.date_input("Enter date")
store_nbr = st.number_input("Enter store number")
family = st.selectbox("Choose item family",('AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY', 'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'))     
on_promotion = st.selectbox('Is the item on promotion?',('0 - No', '1 - Yes'))
city = st.selectbox("Enter city",('Ambato', 'Babahoyo', 'Cayambe', 'Cuenca', 'Daule', 'El Carmen', 'Esmeraldas', 'Guaranda', 'Guayaquil', 'Ibarra', 'Latacunga', 'Libertad', 'Loja', 'Machala', 'Manta', 'Playas', 'Puyo', 'Quevedo', 'Quito', 'Riobamba', 'Salinas', 'Santo Domingo'))
state = st.selectbox("Enter state",('Azuay', 'Bolivar', 'Chimborazo', 'Cotopaxi', 'El Oro', 'Esmeraldas', 'Guayas', 'Imbabura', 'Loja', 'Los Rios', 'Manabi', 'Pastaza', 'Pichincha', 'Santa Elena', 'Santo Domingo de los Tsachilas', 'Tungurahua')) 
type_x = st.selectbox('type_x',('Holiday', 'Additional', 'Transfer', 'Event', 'Bridge'))
cluster = st.selectbox('Choose cluster',(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
oil_price = st.number_input("Enter oil price")
type_y = st.selectbox('type_y',('A', 'B', 'C', 'D', 'E'))

# Prediction button
if st.button('Predict'):
    # Dataframe creation 
    df = pd.DataFrame(
        {
        "date": [date], "store_nbr": [store_nbr], "family": [family], "promotion": [on_promotion], "city": [city], "state": [state], "type_x": [type_x], "cluster": [cluster ], "oil_price": [oil_price], "type_y": [type_y],
        }
    )
    print(f"[Info] Input data as dataframe :\n {df.to_markdown()}")
    
 

 