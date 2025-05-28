import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from archer.environment import BatchedMDDialEnv, DISEASE_SYMPTOMS
import os

def load_trained_model(model_path=r"Model-145\trainer.pt", device='cuda'):
    try:
        print(f"Attempting to load base GPT-2 model and tokenizer from cache (if available) or Hugging Face Hub...")
        model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Base GPT-2 model and tokenizer initialized successfully.")

        print(f"Now attempting to load *trained* weights from: {model_path}")
        trainer_state = torch.load(model_path, map_location=device)
        if 'model_state_dict' in trainer_state:
            state_dict = trainer_state['model_state_dict']
        elif 'model' in trainer_state:
            state_dict = trainer_state['model']
        else:
            state_dict = trainer_state

        if not state_dict:
            raise ValueError(f"State dictionary from {model_path} is empty or invalid.")

        model.load_state_dict(state_dict)
        print(f"Successfully loaded *trained* model weights from: {model_path}")

    except Exception as e:
        # Đây là dòng cực kỳ quan trọng để bạn thấy lỗi khi load trainer.pt
        print(f"!!!!! CRITICAL ERROR: Failed to load trained model from {model_path}. Error: {str(e)} !!!!!")
        return None, None # Trả về None để hàm gọi có thể kiểm tra

    return model, tokenizer

def create_env(device='cuda', bsize=1, env_load_path=''):
    # Use the user's home directory instead of root's
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    
    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    return BatchedMDDialEnv(
        device=device,
        max_conversation_length=20,
        bsize=bsize,
        env_load_path=env_load_path,
        cache_dir=cache_dir
    )

def generate_question(model, tokenizer, history, device='cuda', curr_disease=None, question_count=0, asked_questions=None):
    if asked_questions is None:
        asked_questions = set()
        
    # Format prompt for asking a question or making a diagnosis
    prompt = f"""You are a medical doctor conducting a patient interview. Based on the conversation history, either:
1. Ask ONE relevant question about the patient's symptoms, or
2. If you have enough information, make a diagnosis.

IMPORTANT: 
- Only ask ONE question at a time. Do not ask multiple questions.
- You must ask at least 5 questions before making a diagnosis.
- If making a diagnosis, it MUST be one of these diseases: {', '.join(DISEASE_SYMPTOMS.keys())}
- Ask about different symptoms each time, don't repeat questions.
- Be systematic in your questioning - ask about different body systems.
- DO NOT repeat any of these already asked questions: {', '.join(asked_questions) if asked_questions else 'None'}
"""

    # Add current disease symptoms if available
    if curr_disease and curr_disease in DISEASE_SYMPTOMS:
        prompt += f"\n- Consider the following symptoms when making a diagnosis:\n  {', '.join(DISEASE_SYMPTOMS[curr_disease])}"

    prompt += f"""

Conversation history:
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
    
    # Decode and process output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Doctor: " in response:
        response = response.split("Doctor: ")[-1]
    
    response = response.strip()
    
    # If it's a diagnosis, check if we've asked enough questions
    if response.lower().startswith(("based on your symptoms", "i diagnose you with", "you have")):
        if question_count < 5:
            # Force the model to ask another question instead of diagnosing
            return generate_question(model, tokenizer, history, device, curr_disease, question_count, asked_questions)
        diagnoses = response.split("Based on your symptoms")
        if len(diagnoses) > 1:
            first_diagnosis = "Based on your symptoms" + diagnoses[1]
            if "." in first_diagnosis:
                first_diagnosis = first_diagnosis.split(".")[0] + "."
            return first_diagnosis.strip()
    
    # If it's a question, take only the first one and check for repetition
    if "?" in response:
        questions = response.split("?")
        response = questions[0] + "?"
        
        # Check if this question has been asked before
        if response in asked_questions:
            # If it's a repeat, generate a new question
            return generate_question(model, tokenizer, history, device, curr_disease, question_count, asked_questions)
        
        # Add the new question to asked_questions
        asked_questions.add(response)
    
    return response.strip()

def run_demo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_trained_model(device=device)
    if model is None or tokenizer is None:
        print("ERROR: Main trained model or tokenizer could not be loaded. Please fix the error above and try again.")
        return # Thoát nếu model chính không tải được

    env = create_env(device=device, bsize=1, env_load_path=r"mddial_t5_base_oracle.pt")
    
    # Reset environment
    history = env.reset()[0]
    print("Initial state:", history)
    
    question_count = 0
    asked_questions = set()
    
    # Run conversation turns
    for _ in range(20):
        # Generate question or diagnosis
        response = generate_question(model, tokenizer, history, device, curr_disease=env.env_list[0].curr_disease, 
                                   question_count=question_count, asked_questions=asked_questions)
        
        # If the response is a diagnosis, use it
        if response.lower().startswith(("based on your symptoms", "i diagnose you with", "you have")):
            diagnosis = response
            break
            
        print("Doctor:", response)
        question_count += 1
        
        # Environment responds
        history, reward, done = env.step([response])[0]
        print(history.split("\n")[-2])
        
        if done:
            break
    
    # If we haven't made a diagnosis yet, ask the model to make one
    if 'diagnosis' not in locals():
        diagnosis = generate_question(model, tokenizer, history, device, curr_disease=env.env_list[0].curr_disease, 
                                    question_count=question_count, asked_questions=asked_questions)
        # if diagnosis.lower().startswith(("based on your symptoms", "i diagnose you with", "you have")):
        #     diagnoses = diagnosis.split("Based on your symptoms")
        #     if len(diagnoses) > 1:
        #         diagnosis = "Based on your symptoms" + diagnoses[1]
        #         if "." in diagnosis:
        #             diagnosis = diagnosis.split(".")[0] + "."
    
    print("\nFinal diagnosis:", diagnosis)
    print("Correct diagnosis:", env.env_list[0].curr_disease)
    
    # Check result
    history, reward, done = env.diagnose_batch([diagnosis])[0]
    print("\nReward:", reward)

if __name__ == "__main__":
    run_demo()