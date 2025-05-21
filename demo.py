import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from archer.environment import BatchedMDDialEnv, DISEASE_SYMPTOMS
import os

def load_model(model_path="/home/biggod/archer/Model-145/trainer.pt", device='cuda'):
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    try:
        trainer_state = torch.load(model_path, map_location=device)
        if 'model_state_dict' in trainer_state:
            state_dict = trainer_state['model_state_dict']
        elif 'model' in trainer_state:
            state_dict = trainer_state['model']
        else:
            state_dict = trainer_state
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    
    return model, tokenizer

def create_env(device='cuda', bsize=1, env_load_path=''):
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    os.makedirs(cache_dir, exist_ok=True)
    
    return BatchedMDDialEnv(
        device=device,
        max_conversation_length=20,
        bsize=bsize,
        env_load_path=env_load_path,
        cache_dir=cache_dir
    )

def generate_response(model, tokenizer, history, device='cuda', curr_disease=None, asked_questions=None):
    if asked_questions is None:
        asked_questions = set()
    
    # Minimal prompt focusing on the conversation flow
    prompt = f"""Conversation:
{history}

Doctor:"""
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Doctor: " in response:
        response = response.split("Doctor: ")[-1]
    
    response = response.strip()
    
    # Handle diagnosis
    if response.lower().startswith(("based on your symptoms", "i diagnose you with", "you have")):
        if len(asked_questions) < 5:
            return generate_response(model, tokenizer, history, device, curr_disease, asked_questions)
        # Take only the first diagnosis
        diagnoses = response.split("Based on your symptoms")
        if len(diagnoses) > 1:
            first_diagnosis = "Based on your symptoms" + diagnoses[1]
            if "." in first_diagnosis:
                first_diagnosis = first_diagnosis.split(".")[0] + "."
            return first_diagnosis.strip()
        return response
    
    # Handle questions
    if "?" in response:
        questions = response.split("?")
        response = questions[0] + "?"
        
        if response in asked_questions:
            return generate_response(model, tokenizer, history, device, curr_disease, asked_questions)
        
        asked_questions.add(response)
    
    return response.strip()

def clean_diagnosis(diagnosis):
    """Helper function to clean up diagnosis response"""
    if diagnosis.lower().startswith(("based on your symptoms", "i diagnose you with", "you have")):
        diagnoses = diagnosis.split("Based on your symptoms")
        if len(diagnoses) > 1:
            diagnosis = "Based on your symptoms" + diagnoses[1]
            if "." in diagnosis:
                diagnosis = diagnosis.split(".")[0] + "."
    return diagnosis.strip()

def run_demo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(device=device)
    env = create_env(device=device, bsize=1, env_load_path="/home/biggod/archer/mddial_t5_base_oracle.pt")
    
    history = env.reset()[0]
    print("Initial state:", history)
    
    asked_questions = set()
    
    for _ in range(20):
        response = generate_response(model, tokenizer, history, device, 
                                  curr_disease=env.env_list[0].curr_disease, 
                                  asked_questions=asked_questions)
        
        if response.lower().startswith(("based on your symptoms", "i diagnose you with", "you have")):
            diagnosis = clean_diagnosis(response)
            break
            
        print("Doctor:", response)
        
        history, reward, done = env.step([response])[0]
        print(history.split("\n")[-2])
        
        if done:
            break
    
    if 'diagnosis' not in locals():
        diagnosis = generate_response(model, tokenizer, history, device, 
                                    curr_disease=env.env_list[0].curr_disease, 
                                    asked_questions=asked_questions)
        diagnosis = clean_diagnosis(diagnosis)
    
    print("\nFinal diagnosis:", diagnosis)
    print("Correct diagnosis:", env.env_list[0].curr_disease)
    
    history, reward, done = env.diagnose_batch([diagnosis])[0]
    print("\nReward:", reward)

if __name__ == "__main__":
    run_demo() 