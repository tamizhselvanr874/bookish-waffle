import streamlit as st  
import pandas as pd  
import openai  
  
# Configuration  
API_KEY = "783973291a7c4a74a1120133309860c0"  
ENDPOINT = "https://theswedes.openai.azure.com/"  
API_VERSION = "2024-05-01-preview"  
DEPLOYMENT_NAME = "GPT-4-Omni"  
  
# Initialize OpenAI client  
openai.api_type = "azure"  
openai.api_key = API_KEY  
openai.api_base = ENDPOINT  
openai.api_version = API_VERSION  
  
# Define prompt templates for each technique  
def generate_prompt(method, context, goal):  
    prompts = {  
        "PCP": f"Progressive Contextual Prompting (PCP):\nContext: {context}\nGoal: {goal}\nEnsure the response builds on previous context, addresses the goal, and maintains engagement.",  
        "Zero-shot": f"Zero-shot Prompting:\nTask: {goal}\nContext: {context}\n",  
        "Few-shot": f"Few-shot Prompting:\nExamples: Provide examples relevant to the task.\nContext: {context}\nGoal: {goal}\n",  
        "CoT": f"Chain-of-Thought (CoT) Prompting:\nContext: {context}\nGoal: {goal}\nReasoning: Break down the problem into logical steps.",  
        "Meta": f"Meta Prompting:\nStructure: Focus on format and pattern over specific content.\nContext: {context}\nGoal: {goal}\n",  
        "Self-Consistency": f"Self-Consistency Prompting:\nContext: {context}\nGoal: {goal}\nGenerate multiple reasoning paths and choose the most consistent answer.",  
        "ReAct": f"Reason and Act (ReAct) Prompting:\nContext: {context}\nGoal: {goal}\nCombine reasoning and actions for problem-solving.",  
        "Tree-of-Thoughts": f"Tree-of-Thoughts (ToT) Prompting:\nContext: {context}\nGoal: {goal}\nExplore multiple reasoning paths and evaluate decisions."  
    }  
    return prompts.get(method, "")  
  
def evaluate_prompt(method, context, goal):  
    prompt = generate_prompt(method, context, goal)  
    try:  
        response = openai.ChatCompletion.create(  
            engine=DEPLOYMENT_NAME,  
            messages=[  
                {"role": "system", "content": "You are an AI assistant."},  
                {"role": "user", "content": prompt}  
            ],  
            max_tokens=512,  
            temperature=0.5,  
            top_p=0.9,  
            frequency_penalty=0,  
            presence_penalty=0,  
        )  
        return response.choices[0].message['content'].strip()  
    except Exception as e:  
        st.error(f"Error: {str(e)}")  
        return ""  
  
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
    problem = st.selectbox(  
        "Problem",   
        ["Analyze customer data", "Visualize data trends", "Optimize supply chain", "Other problem..."]  
    )  
  
    st.write("### Define the goal of the problem:")  
    goal = st.text_input("Goal", "Identify key trends")  
  
    st.write("### Select prompting techniques to compare:")  
    techniques = st.multiselect(  
        "Techniques",   
        ["PCP", "Zero-shot", "Few-shot", "CoT", "Meta", "Self-Consistency", "ReAct", "Tree-of-Thoughts"]  
    )  
  
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
  
        st.write("### Evaluation Results:")  
        st.dataframe(results_df[["Technique", "Length", "Coherence", "Relevance", "Completeness"]])  
  
        st.write("### Detailed Responses:")  
        for index, row in results_df.iterrows():  
            st.write(f"**{row['Technique']}**: {row['Response']}")  
  
if __name__ == "__main__":  
    main()  
