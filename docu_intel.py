import os  
import streamlit as st  
import pandas as pd  
import openai  
  
# Set up environment variables  
os.environ["AZURE_OPENAI_API_KEY"] = "783973291a7c4a74a1120133309860c0"  
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://theswedes.openai.azure.com/"  
os.environ["OPENAI_API_TYPE"] = "azure"  
os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"  
os.environ["AZURE_DEPLOYMENT_NAME"] = "GPT-4-Omni"  
  
# Configuration  
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
API_VERSION = os.getenv("OPENAI_API_VERSION")  
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")  
  
# Initialize OpenAI client  
openai.api_type = "azure"  
openai.api_key = API_KEY  
openai.api_base = ENDPOINT  
openai.api_version = API_VERSION  
  
# Define prompt templates for each technique  
def generate_prompt(method, context, goal):  
    if method == "PCP":  
        return (  
            f"Progressive Contextual Prompting (PCP):\n"  
            f"Context: {context}\n"  
            f"Goal: {goal}\n"  
            f"Ensure the response builds on previous context, addresses the goal, and maintains engagement."  
        )  
    elif method == "Zero-shot":  
        return (  
            f"Zero-shot Prompting:\n"  
            f"Task: {goal}\n"  
            f"Context: {context}\n"  
        )  
    elif method == "Few-shot":  
        return (  
            f"Few-shot Prompting:\n"  
            f"Examples: Provide examples relevant to the task.\n"  
            f"Context: {context}\n"  
            f"Goal: {goal}\n"  
        )  
    elif method == "CoT":  
        return (  
            f"Chain-of-Thought (CoT) Prompting:\n"  
            f"Context: {context}\n"  
            f"Goal: {goal}\n"  
            f"Reasoning: Break down the problem into logical steps."  
        )  
    elif method == "Meta":  
        return (  
            f"Meta Prompting:\n"  
            f"Structure: Focus on format and pattern over specific content.\n"  
            f"Context: {context}\n"  
            f"Goal: {goal}\n"  
        )  
    elif method == "Self-Consistency":  
        return (  
            f"Self-Consistency Prompting:\n"  
            f"Context: {context}\n"  
            f"Goal: {goal}\n"  
            f"Generate multiple reasoning paths and choose the most consistent answer."  
        )  
    elif method == "ReAct":  
        return (  
            f"Reason and Act (ReAct) Prompting:\n"  
            f"Context: {context}\n"  
            f"Goal: {goal}\n"  
            f"Combine reasoning and actions for problem-solving."  
        )  
    elif method == "Tree-of-Thoughts":  
        return (  
            f"Tree-of-Thoughts (ToT) Prompting:\n"  
            f"Context: {context}\n"  
            f"Goal: {goal}\n"  
            f"Explore multiple reasoning paths and evaluate decisions."  
        )  
  
def evaluate_prompt(method, context, goal):  
    prompt = generate_prompt(method, context, goal)  
    response = openai.ChatCompletion.create(  
        deployment_id=DEPLOYMENT_NAME,  
        messages=[  
            {"role": "system", "content": "You are an AI assistant designed to evaluate prompting techniques."},  
            {"role": "user", "content": prompt}  
        ],  
        max_tokens=512,  
        temperature=0.5,  
        top_p=0.9,  
        frequency_penalty=0,  
        presence_penalty=0,  
    )  
    return response.choices[0].message['content'].strip()  
  
def evaluate_quality(response):  
    # Implement real evaluation metrics here  
    return {  
        "Length": len(response.split()),  
        "Coherence": 0.8,  # Placeholder score  
        "Relevance": 0.9,  # Placeholder score  
        "Completeness": 0.85  # Placeholder score  
    }  
  
def main():  
    st.title("Advanced Prompting Techniques Evaluation")  
  
    st.write("### Select a problem to evaluate:")  
    problem = st.selectbox("Problem", ["Analyze customer data", "Visualize data trends", "Other problem..."])  
  
    st.write("### Define the goal of the problem:")  
    goal = st.text_input("Goal", "Identify key trends")  
  
    st.write("### Select prompting techniques to compare:")  
    techniques = st.multiselect("Techniques", ["PCP", "Zero-shot", "Few-shot", "CoT", "Meta", "Self-Consistency", "ReAct", "Tree-of-Thoughts"])  
  
    context = st.text_area("Provide initial context for the problem:")  
  
    if st.button("Evaluate"):  
        results = []  
        for technique in techniques:  
            response = evaluate_prompt(technique, context, goal)  
            quality_scores = evaluate_quality(response)  
            results.append({  
                "Technique": technique,  
                "Response": response,  
                "Length": quality_scores["Length"],  
                "Coherence": quality_scores["Coherence"],  
                "Relevance": quality_scores["Relevance"],  
                "Completeness": quality_scores["Completeness"]  
            })  
  
        # Sort results by the evaluation scores  
        results_df = pd.DataFrame(results).sort_values(by=["Coherence", "Relevance", "Completeness"], ascending=False)  
  
        # Highlight PCP  
        results_df.style.apply(lambda x: ['background: yellow' if x['Technique'] == 'PCP' else '' for i in x], axis=1)  
  
        st.write("### Evaluation Results:")  
        st.dataframe(results_df[["Technique", "Length", "Coherence", "Relevance", "Completeness"]])  
  
        st.write("### Detailed Responses:")  
        for index, row in results_df.iterrows():  
            st.write(f"**{row['Technique']}**: {row['Response']}")  
  
if __name__ == "__main__":  
    main()  
