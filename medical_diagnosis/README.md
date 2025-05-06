# Medical Diagnosis Workflow using Langgraph

This project demonstrates a multi-agent medical diagnosis workflow using Langgraph. The workflow simulates a medical diagnosis process where multiple specialist agents collaborate to diagnose a patient and create a treatment plan.

## Workflow Components

1. **Patient Symptoms Generator**: Generates realistic patient symptoms and medical history
2. **General Practitioner**: Provides initial assessment and recommends specialists
3. **Specialist Agents**:
   - Cardiologist
   - Neurologist
   - Pulmonologist
4. **Final Diagnosis Agent**: Consolidates all opinions and provides final diagnosis and treatment plan

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the workflow:
```bash
python medical_workflow.py
```

The workflow will:
1. Generate patient symptoms
2. Get initial assessment from the general practitioner
3. Consult with specialists
4. Provide a final diagnosis and treatment plan

## Output

The workflow will output:
- Generated patient symptoms
- Initial diagnosis from the general practitioner
- Opinions from each specialist
- Final diagnosis and comprehensive treatment plan

## Note

This is a simulation and should not be used for actual medical diagnosis. The agents are powered by language models and should be used for educational purposes only. 