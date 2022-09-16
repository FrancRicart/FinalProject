import numpy as np
import pandas as pd

import statsmodels.stats.api as sms
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st 
from statsmodels.graphics.mosaicplot import mosaic


df = pd.read_csv('../data/Admission_Predict_ver1.1.csv')
treshold = pd.read_csv('../data/treshold75.csv')
HighChance = pd.read_csv('../data/HighChance.csv')


st.set_page_config(
	page_title = "Ivy League Forecaster",
	page_icon="👨‍🎓"
	)

#st.dataframe(data=df)

########
def gre_plot(data):
    fig1, ax = plt.subplots(figsize=(15, 10))
    sns.distplot(df["GRE Score"],bins=25,ax=ax)
    sns.despine()
    std = df["GRE Score"].std()
    mean = df["GRE Score"].mean()

    #Limit lines
    plt.axvline(mean+std,  label='One standard deviation(+)',c='#30a2da')
    plt.axvline(mean-std,  label='One standard deviation(-)',c='#30a2da')

    plt.axvline(mean+2*std,  label='Two standard deviation(+)',c='#fc4f30')
    plt.axvline(mean-2*std,  label='Two standard deviation(-)',c='#fc4f30')

    plt.axvline(317,  label='Good Score',c='#6d904f')
    plt.axvline(328,  label='Excellent Score',c='#e5ae38')

    #Title and legend
    #plt.title("GRE Score Distribution(with Empirical Rule)")
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3, fancybox=True, shadow=True);
    
    return fig1
 ######

######
def toefl_plot(data):
	fig2, ax = plt.subplots(figsize=(15, 10))
	sns.distplot(df["TOEFL Score"],bins=25,ax=ax)
	sns.despine()
	std = df["TOEFL Score"].std()

	#Title and legend
	#plt.title("TOEFL Score Distribution(with Empirical Rule)")
	#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3, fancybox=True, shadow=True);mean = data.TOEFL.mean()


	#Limit lines
	#plt.axvline(mean+std,  label='One standard deviation(+)',c='#30a2da')
	#plt.axvline(mean-std,  label='One standard deviation(-)',c='#30a2da')

	#plt.axvline(mean+2*std,  label='Two standard deviation(+)',c='#fc4f30')
	#plt.axvline(mean-2*std,  label='Two standard deviation(-)',c='#fc4f30')

	plt.axvline(90,  label='Good Score',c='#6d904f')
	plt.axvline(110,  label='Excellent Score',c='#e5ae38')

	return fig2
	######

#####
def sop_plot(data):
	fig3, ax = plt.subplots(figsize=(15, 10))
	sns.distplot(df["TOEFL Score"],bins=10,ax=ax)
	sns.despine()
	std = df["TOEFL Score"].std()
	mean = df["TOEFL Score"].mean()

	#Title and legend
	#plt.title("SOP Distribution(with Empirical Rule)")
	#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fancybox=True, shadow=True);


	#Limit lines
	plt.axvline(mean+std,  label='One standard deviation(+)',c='#30a2da')
	plt.axvline(mean-std,  label='One standard deviation(-)',c='#30a2da')

	plt.axvline(mean+2*std,  label='Two standard deviation(+)',c='#fc4f30')
	plt.axvline(mean-2*std,  label='Two standard deviation(-)',c='#fc4f30')

	return fig3
#####

#####
def lor_plot(data):
	fig4, ax = plt.subplots(figsize=(15, 10))
	sns.distplot(df["LOR "],bins=10,ax=ax)
	sns.despine()
	std = df["LOR "].std()
	mean = df["LOR "].mean()

	#Title and legend
	#plt.title("LOR Distribution(with Empirical Rule)")
	#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fancybox=True, shadow=True);

	#Limit line
	plt.axvline(mean+std,  label='One standard deviation(+)',c='#30a2da')
	plt.axvline(mean-std,  label='One standard deviation(-)',c='#30a2da')

	plt.axvline(mean+2*std,  label='Two standard deviation(+)',c='#fc4f30')
	plt.axvline(mean-2*std,  label='Two standard deviation(-)',c='#fc4f30')


	return fig4		
#####

#####
def cgpa_plot(data):
	fig5, ax = plt.subplots(figsize=(15, 10))
	sns.distplot(df["CGPA"],bins=10,ax=ax)
	sns.despine()
	std = df["CGPA"].std()
	mean = df["CGPA"].mean()

	#Limit line
	plt.axvline(mean+std,  label='One standard deviation(+)',c='#30a2da')
	plt.axvline(mean-std,  label='One standard deviation(-)',c='#30a2da')

	plt.axvline(mean+2*std,  label='Two standard deviation(+)',c='#fc4f30')
	plt.axvline(mean-2*std,  label='Two standard deviation(-)',c='#fc4f30')

	#Title and legend
	#plt.title("CGPA Distribution(with Empirical Rule)")
	#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fancybox=True, shadow=True);

	return fig5
#####

#####
def chance_plot(data):
	fig6, ax = plt.subplots(figsize=(15, 10))
	sns.distplot(df["Chance of Admit "],bins=20,ax=ax)
	sns.despine()
	std = df["Chance of Admit "].std()
	mean = df["Chance of Admit "].mean()

	#Limit lines
	plt.axvline(mean+std,  label='One standard deviation(+)',c='#30a2da')
	plt.axvline(mean-std,  label='One standard deviation(-)',c='#30a2da')

	plt.axvline(mean+2*std,  label='Two standard deviation(+)',c='#fc4f30')
	plt.axvline(mean-2*std,  label='Two standard deviation(-)',c='#fc4f30')

	#Title and legend
	#plt.title("Acceptance Percentage (with Empirical Rule)")
	#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fancybox=True, shadow=True);

	return fig6
#####

#####
def uni_rating_chances(data):
	HighChance["Research"] = HighChance["Research"].map({1:'Research',0:'No Research'})
	HighChance["High_Chance"] = HighChance["High_Chance"].map({1:'Accepted',0:'Rejected'})

	fig7, ax = plt.subplots(figsize=(15, 10))
	mosaic(HighChance.sort_values('UnivRating'),['UnivRating','High_Chance'], gap=0.01, title='How University Rating contribute in getting you accepted?',ax=ax);

	return fig7
####

#####
def research_chances(data):

	fig8, ax = plt.subplots(figsize=(15, 10))
	mosaic(HighChance,['Research','High_Chance'], gap=0.05, title='Can Research to get you accepted?',ax=ax);

	return fig8
######

#####
def toefl_sop(data):
	fig9, ax = px.box(data, 
             x="SOP", 
             y="TOEFL Score",
             color="SOP", 
             points="all",
             color_discrete_sequence=px.colors.sequential.RdBu,
             title="Can good TOEFL help you get good SOP Score?",
             
            )
	fig9.update_layout(legend={"itemclick":False})
	fig9.update_layout(legend={"itemdoubleclick":False})
	fig9.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

	fig9.update_layout(plot_bgcolor="rgb(255,255,255)")

	return fig9
######


#####
def topuni_sop(data):
	fig10 = sns.catplot(x="University Rating", y="SOP", kind="violin", data=data, height=8, palette = "pastel")
	#plt.title('Does being from Top Universities benefit candidates in SOP score?');

	return fig10
#####

#####
def topuni_lor(data):
	fig11 = sns.catplot(x="University Rating", y="LOR ", kind="violin", data=data, height=10, palette = "pastel" )
	
	return fig11
#####

#####
def topuni_research(data):
	fig12 = sns.catplot(x="Research", y="University Rating", kind="boxen", data=data, height=8, palette="pastel")

	
	return fig12
#####

#####
def reseach_LOR(data):
	fig13 = sns.catplot(x="Research", y="LOR ", kind="boxen", data=data, height=8, palette= "pastel")

	return fig13 

##### 




#####

st.title("Insights of the App")

st.markdown("In order to understad a little bit more how this app works under the hood, maybe a quick glance to the data taken into account might help.")
st.markdown("First, let's see the distribution of the different tests and evaluations that are considered.")


st.sidebar.success("Select a page above")

#st.markdown("<h1 style='text-align: center; color: black;'>Hello World!</h1>", unsafe_allow_html=True)


c1 = st.container()
with c1:

	col1, col2, col3 = st.columns(3)
	
	with col1:
		st.subheader("Distribution of GRE")
		gre_plot = gre_plot(df)
		st.pyplot(gre_plot)

	with col2:
		st.subheader("Distribution of TOEFL")
		toefl_plot = toefl_plot(df)
		st.pyplot(toefl_plot)

	with col3:
		st.subheader("Distribution of SOP")
		sop_plot = sop_plot(df)
		st.pyplot(sop_plot)



c2 = st.container()
with c2:

	col1, col2, col3 = st.columns(3)
	
	with col1:
		st.subheader("Distribution of LOR")
		lor_plot = lor_plot(df)
		st.pyplot(lor_plot)

	with col2:
		st.subheader("Distribution of CGPA")
		cgpa_plot = cgpa_plot(df)
		st.pyplot(cgpa_plot)

	with col3:
		st.subheader("CHANCE OF BEING ADMITTED")
		chance_plot = chance_plot(df)
		st.pyplot(chance_plot)



st.header("Now let's try to get some interesting info from the data")


c3 = st.container()
with c3:

	col1, col2 = st.columns(2)

	with col1:
		st.subheader("Are high CGPA Scores candidates more keen for research?")
		fig1 = px.box(df, x="Research", y="CGPA", points="all", color = "Research", 
		             color_discrete_sequence=px.colors.sequential.RdBu, 
		             #hover_data = ({data["research"]:False}),
		             #labels = {"0": "No Reseach", "1":"Research"},
		             #title="Do high CGPA scores candidate more keen for Research?",
		             #hoverinfo = None
		            )


		fig1.update_layout(legend={"itemclick":False})
		fig1.update_layout(legend={"itemdoubleclick":False})
		fig1.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
		fig1.update_layout(plot_bgcolor="rgb(255,255,255)")

		st.plotly_chart(fig1)
		

	with col2:
		st.subheader("Do high GRE Scores candidate more keen for research?")
 
		fig2 = px.box(df, 
		             x="Research", 
		             y="GRE Score", 
		             points="all", 
		             color="Research", 
		             color_discrete_sequence=px.colors.sequential.RdBu, 
		             #hover_data = ({data["research"]:False}),
		             #labels = {"0": "No Reseach", "1":"Research"},
		             #title="Do high GRE scores candidate more keen for Research?",
		             #hoverinfo = None
		            )
		fig2.update_layout(legend={"itemclick":False})
		fig2.update_layout(legend={"itemdoubleclick":False})
		fig2.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
		fig2.update_layout(plot_bgcolor="rgb(255,255,255)")

		st.plotly_chart(fig2)


st.markdown("We can observe an interesting correlation between higher CGPA and GRE scores with inclination to do research!")

c4 = st.container()
with c4:

	col1, col2 = st.columns(2)

	with col1:
		st.subheader("Can good TOEFL help you get good SOP Score?")
 
		fig9 = px.box(df, 
             x="SOP", 
             y="TOEFL Score",
             color="SOP", 
             points="all",
             color_discrete_sequence=px.colors.sequential.RdBu,
             title="Can good TOEFL help you get good SOP Score?",
             
            )
		fig9.update_layout(legend={"itemclick":False})
		fig9.update_layout(legend={"itemdoubleclick":False})
		fig9.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

		fig9.update_layout(plot_bgcolor="rgb(255,255,255)")
		
		st.plotly_chart(fig9)



	with col2:
		st.subheader("Acceptance threshold 75%")
		donut = px.pie(treshold, values='Count', names='Outcome', color_discrete_sequence=px.colors.sequential.RdBu,hole=.7,title='GRE Acceptance Distribution',width=600,height=600)\
		.update_traces(textposition='inside', textinfo='percent+label')\
		.update_layout(legend={"itemclick":False})\
		.update_layout(legend={"itemdoubleclick":False})\
		.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

		st.plotly_chart(donut)


st.markdown("Here a threshold for the top quantile to asseverated those who more surely are going to be admitted is stablished.")
st.markdown("This is a very interesting approach to study in the next 2 plots: Your University Ranking and the fact that you do reaearch are goint to gramatically influence your chances to be admitted in an IVY University!")


c10 = st.container()
with c10:
	col1, col2 = st.columns(2)

	with col1:
		st.subheader("Can Uni Rating Increment or Decrement your chances of getting accepted?")

		fig7 = uni_rating_chances(HighChance)
		st.pyplot(fig7)

	with col2:
		st.subheader("Can Research help you getting accepted?")

		fig8 = research_chances(HighChance)
		st.pyplot(fig8)




c11 = st.container()
with c11:
	col1, col2 = st.columns(2)

	with col1:
		st.subheader("Does being from Top Universities benefit candidates in SOP score?")

		fig10 = topuni_sop(df)
		st.pyplot(fig10)

	with col2:
		st.subheader("Does being in Top Universities benefit you in LOR score?")

		fig11 = topuni_lor(df)
		st.pyplot(fig11)



c12 = st.container()
with c12:
	col1, col2 = st.columns(2)

	with col1:
		st.subheader("Do top Unis encourage Research?")

		fig12 = topuni_research(df)
		st.pyplot(fig12)

	with col2:
		st.subheader("Can Research help in getting higher LOR Score?")

		fig13 = reseach_LOR(df)
		st.pyplot(fig13)


st.markdown("Are you ready to see what are your chances to be admitted? Give it a try on our **IVY FORESCASTER**!! Best of lucks!!👉")


