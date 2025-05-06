from typing import Dict, List, Tuple, Annotated
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from air_genai_helper import initialize_urls_and_key
import time
from typing import Any, Dict, List
import operator
import logging
from datetime import datetime

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/medical_workflow_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return log_file

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI
vault_url, api_key, azure_endpoint = initialize_urls_and_key("ts", "eastus2")

# Custom dictionary merge operator
def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    return {**dict1, **dict2}

# Custom operator to take the latest value
def take_latest(val1: str, val2: str) -> str:
    return val2

class TokenTracker(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def update(self, response: Any):
        if hasattr(response, 'usage_metadata'):
            self.total_tokens += response.usage_metadata['total_tokens']
            self.prompt_tokens += response.usage_metadata['input_tokens']
            self.completion_tokens += response.usage_metadata['output_tokens']

def init_llm():
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # or your deployment
        api_version="2024-06-01",
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )
    return llm

# Get LLM instance
llm = init_llm()
token_tracker = TokenTracker()

# Define our state
class MedicalState(BaseModel):
    patient_symptoms: Annotated[str, take_latest] = Field(default="")
    current_diagnosis: Annotated[str, operator.add] = Field(default="")
    treatment_plan: Annotated[List[str], operator.add] = Field(default_factory=list)
    specialist_opinions: Annotated[Dict[str, str], merge_dicts] = Field(default_factory=dict)
    final_diagnosis: Annotated[str, operator.add] = Field(default="")
    final_treatment: Annotated[str, operator.add] = Field(default="")
    execution_time: Annotated[float, operator.add] = Field(default=0.0)

# Patient Symptoms Generator Agent
def generate_patient_symptoms(state: MedicalState) -> MedicalState:
    prompt = """Generate a realistic set of patient symptoms and medical history. 
    Include age, gender, main symptoms, duration of symptoms, and any relevant medical history.
    Make it detailed but not too complex."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    token_tracker.update(response)
    # Only take the first occurrence of the patient information
    symptoms = response.content.split("**Patient Information**")[0]
    state.patient_symptoms = symptoms
    logging.info("Generated patient symptoms")
    return state

# General Practitioner Agent
def general_practitioner(state: MedicalState) -> MedicalState:
    prompt = f"""As a General Practitioner, analyze these patient symptoms and provide an initial assessment:
    {state.patient_symptoms}
    
    Provide:
    1. Initial diagnosis
    2. Recommended specialists to consult
    3. Any immediate concerns"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    token_tracker.update(response)
    state.current_diagnosis = response.content
    logging.info("General practitioner assessment completed")
    return state

# Specialist Agents
def cardiologist(state: MedicalState) -> MedicalState:
    prompt = f"""As a Cardiologist, review the patient's symptoms and current diagnosis:
    Symptoms: {state.patient_symptoms}
    Current Diagnosis: {state.current_diagnosis}
    
    Provide your expert opinion on any cardiac concerns and recommended treatments."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    token_tracker.update(response)
    state.specialist_opinions["cardiologist"] = response.content
    logging.info("Cardiologist consultation completed")
    return state

def neurologist(state: MedicalState) -> MedicalState:
    prompt = f"""As a Neurologist, review the patient's symptoms and current diagnosis:
    Symptoms: {state.patient_symptoms}
    Current Diagnosis: {state.current_diagnosis}
    
    Provide your expert opinion on any neurological concerns and recommended treatments."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    token_tracker.update(response)
    state.specialist_opinions["neurologist"] = response.content
    logging.info("Neurologist consultation completed")
    return state

def pulmonologist(state: MedicalState) -> MedicalState:
    prompt = f"""As a Pulmonologist, review the patient's symptoms and current diagnosis:
    Symptoms: {state.patient_symptoms}
    Current Diagnosis: {state.current_diagnosis}
    
    Provide your expert opinion on any respiratory concerns and recommended treatments."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    token_tracker.update(response)
    state.specialist_opinions["pulmonologist"] = response.content
    logging.info("Pulmonologist consultation completed")
    return state

# Final Diagnosis and Treatment Plan Agent
def final_diagnosis(state: MedicalState) -> MedicalState:
    prompt = f"""As a senior medical consultant, review all the information and provide a final diagnosis and treatment plan:
    Patient Symptoms: {state.patient_symptoms}
    Initial Diagnosis: {state.current_diagnosis}
    Specialist Opinions: {state.specialist_opinions}
    
    Provide:
    1. Final diagnosis
    2. Comprehensive treatment plan
    3. Follow-up recommendations"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    token_tracker.update(response)
    state.final_diagnosis = response.content
    logging.info("Final diagnosis and treatment plan completed")
    return state

# Create the workflow graph
def create_medical_workflow() -> Graph:
    # Create the graph
    workflow = StateGraph(MedicalState)
    
    # Add nodes
    workflow.add_node("generate_symptoms", generate_patient_symptoms)
    workflow.add_node("general_practitioner", general_practitioner)
    workflow.add_node("cardiologist", cardiologist)
    workflow.add_node("neurologist", neurologist)
    workflow.add_node("pulmonologist", pulmonologist)
    workflow.add_node("final_consultation", final_diagnosis)
    
    # Define the edges
    workflow.add_edge("generate_symptoms", "general_practitioner")
    workflow.add_edge("general_practitioner", "cardiologist")
    workflow.add_edge("general_practitioner", "neurologist")
    workflow.add_edge("general_practitioner", "pulmonologist")
    workflow.add_edge("cardiologist", "final_consultation")
    workflow.add_edge("neurologist", "final_consultation")
    workflow.add_edge("pulmonologist", "final_consultation")
    
    # Set the entry point
    workflow.set_entry_point("generate_symptoms")
    
    return workflow.compile()

# Main execution function
def run_medical_workflow():
    start_time = time.time()
    workflow = create_medical_workflow()
    result_dict = workflow.invoke({})
    end_time = time.time()
    
    # Convert the result dictionary back to MedicalState
    result = MedicalState(**result_dict)
    
    # Add execution time to the result
    result.execution_time = end_time - start_time
    
    return result, token_tracker

if __name__ == "__main__":
    # Set up logging
    log_file = setup_logging()
    logging.info("Starting medical diagnosis workflow")
    
    result, token_tracker = run_medical_workflow()
    
    # Log the results in a more concise way
    logging.info("\n=== Patient Case Summary ===")
    logging.info("Patient Symptoms and History:")
    logging.info(result.patient_symptoms)  # No need to split again since we already did it in generate_patient_symptoms
    
    logging.info("\n=== Medical Assessment ===")
    logging.info("Initial Diagnosis:")
    logging.info(result.current_diagnosis)
    
    logging.info("\n=== Specialist Consultations ===")
    for specialist, opinion in result.specialist_opinions.items():
        logging.info(f"\n{specialist.upper()}:")
        logging.info(opinion)
    
    logging.info("\n=== Final Diagnosis and Treatment Plan ===")
    logging.info(result.final_diagnosis)
    
    # Log performance metrics
    logging.info("\n=== Performance Metrics ===")
    logging.info(f"Total Execution Time: {result.execution_time:.2f} seconds")
    logging.info(f"Total Tokens Used: {token_tracker.total_tokens}")
    logging.info(f"Prompt Tokens: {token_tracker.prompt_tokens}")
    logging.info(f"Completion Tokens: {token_tracker.completion_tokens}")
    
    # Print to console as well
    print("\n=== Patient Case Summary ===")
    print("Patient Symptoms and History:")
    print(result.patient_symptoms)  # No need to split again since we already did it in generate_patient_symptoms
    
    print("\n=== Medical Assessment ===")
    print("Initial Diagnosis:")
    print(result.current_diagnosis)
    
    print("\n=== Specialist Consultations ===")
    for specialist, opinion in result.specialist_opinions.items():
        print(f"\n{specialist.upper()}:")
        print(opinion)
    
    print("\n=== Final Diagnosis and Treatment Plan ===")
    print(result.final_diagnosis)
    
    print("\n=== Performance Metrics ===")
    print(f"Total Execution Time: {result.execution_time:.2f} seconds")
    print(f"Total Tokens Used: {token_tracker.total_tokens}")
    print(f"Prompt Tokens: {token_tracker.prompt_tokens}")
    print(f"Completion Tokens: {token_tracker.completion_tokens}")
    
    logging.info(f"Workflow completed. Log file: {log_file}") 