import pickle
import streamlit as st 
import pandas as pd
import numpy as np
#import sklearn
#from sklearn.linear_model import LinearRegression




st.title("IVY Predictor")

st.header("User Input Parameters")
st.markdown("👉 Introduce your results on the forecaster and obtain your probability of entering into one of the IVY League Unis!")


def user_input_features():
	gre = st.slider("GRE Score", 0, 350, 300)
	toefl = st.slider("TOEFL Score", 0, 130, 100)
	univrating = st.selectbox("University Rating", ("1", "2", "3", "4", "5"))
	sop = st.slider("SOP Score", 0.0, 6.0, 3.5)
	lor = st.slider("LOR Score", 0.0, 6.0, 3.5)
	cgpa = st.slider("CGPA Score", 0.0, 10.0, 8.5)
	research = st.selectbox("Did you do reaserch?", ("Yes", "No"))
	datauser = {"GRE" : [gre],
				"TOEFL": [toefl],
				"UnivRating":[univrating],
				"SOP": [sop],
				"LOR": [lor],
				"CGPA": [cgpa],
				"Research": [research]}
	features = pd.DataFrame(datauser, index=[0])
	features["Research"] = features["Research"].replace("Yes",1).replace("No",0)
	#features["cgpa"] = np.round(features["cgpa"], decimals = 2)
	return features			



df_user = user_input_features()

st.subheader("User Input parameters")
st.markdown("👉 These are your parameters")

st.write(df_user)


with open('../scalers/scaler.pkl', 'rb') as f:
	scaler = pickle.load(f)
	data_scaled = pd.DataFrame(scaler.transform(df_user), columns = df_user.columns)

with open('../transformers/powertransformer.pkl', 'rb') as f:
	transformer = pickle.load(f)
	data_transformed = pd.DataFrame(transformer.transform(data_scaled), columns = df_user.columns)

with open('../models/lm.pkl', 'rb') as f:
 	lm = pickle.load(f)
 	predicted_value = lm.predict(data_transformed)[0]


st.subheader("User Chances")
st.markdown("👉 These are your chances of being admitted!")

st.write("Yor predicted score is **{:2f}**".format(predicted_value))

st.markdown("We hope you the best of lucks with your Admission Process!")
st.markdown("If you liked the **IVY Forecaster** do not hesitate to recomend us to your friends!")

st.image('./Pics/colorful.png')





