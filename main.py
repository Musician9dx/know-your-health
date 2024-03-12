from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import streamlit as st

st.title("Know your health better")

hf1 = HuggingFacePipeline.from_model_id(
    model_id="medalpaca/medalpaca-7b",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 50},
)

hf2=HuggingFacePipeline.from_model_id(
    model_id="AliiaR/DialoGPT-medium-empathetic-dialogues",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 50},
)

prompt1 = PromptTemplate(

    input_variables=["disease"],
    template="what causes {disease}?",
    output_key="disease_details"
)

prompt2 = PromptTemplate(

    input_variables=["disease"],
    template="What are the possible treatments for {disease}?",
    output_key="treatment_details"

)

prompt3 = PromptTemplate(

    input_variables=["disease_details", "treatment_details"],
    template="I am suffering from {disease_details}, doctor told that It can be cured from {treatment_details}",
    output_key="empathy"

)


chain1=LLMChain(llm=hf1,prompt=prompt1, output_key="disease_details")
chain2=LLMChain(llm=hf1,prompt=prompt2, output_key="treatment_details")
chain3=LLMChain(llm=hf2,prompt=prompt3, output_key="empathy")

schain=SequentialChain(
    chains=[chain1,chain2,chain3],
    input_variables=["disease",],
    output_variables=["disease_details","treatment_details","empathy"]
)


text_input=st.text_input("Enter Diseasse Name")


if text_input:

    response=schain({

        "disease": f"{text_input}"

    })

    st.heading(text_input)

    st.subheading("Disease Details")
    st.write(response["disease_details"])

    st.subheading("Treatment Details")
    st.write(response["treatment_details"])

    st.subheading("Don't Worry")
    st.write(response["emapthy"])




