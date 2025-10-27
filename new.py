import pandas as pd
import spacy

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Load your updated dataset CSV file
df = pd.read_csv('remidiAi_finalmerged_dataset_with_causes.csv',encoding='ISO-8859-1')

def get_user_input():
    user_input = input("Enter your symptoms = ")
    user_input = user_input.lower()
    return user_input

# Extract symptom phrases (noun chunks) from user input
def extract_symptom_phrases(user_input):
    doc = nlp(user_input)
    phrases = [chunk.text.strip() for chunk in doc.noun_chunks]
    return phrases

# Match phrases with dataset symptoms based on word overlap
def find_matches_bow(extracted_phrases, dataset_symptoms, threshold=1):
    matched_indices = set()
    for phrase in extracted_phrases:
        phrase_words = set(phrase.split())
        for i, symptom in enumerate(dataset_symptoms):
            symptom_words = set(symptom.split())
            if len(phrase_words.intersection(symptom_words)) >= threshold:
                matched_indices.add(i)
    return matched_indices

def process_input(user_input):
    extracted_phrases = extract_symptom_phrases(user_input)
    dataset_symptoms = df['Symptoms'].str.lower().str.strip().tolist()
    matched_indices = find_matches_bow(extracted_phrases, dataset_symptoms, threshold=1)

    if not matched_indices:
        output = "No matching symptoms found in dataset.\n"
        print(output)
        with open('output_remedies.txt', 'a') as f:
            f.write(output)
    else:
        printed_symptoms = set()
        with open('output_remedies.txt', 'a') as f:
            for idx in matched_indices:
                row = df.iloc[idx]
                symptom = row['Symptoms']
                if symptom not in printed_symptoms:
                    output = (
                        f"Symptoms: {symptom}\n"
                        f"Cause: {row.get('Cause', 'N/A')}\n"
                        f"Chemical Name: {row.get('Chemical_Name', 'N/A')}\n"
                        f"Remedies: {row.get('Remedies', 'N/A')}\n"
                        f"Homemade Soup: {row.get('Homemade_Soup', 'N/A')}\n"
                        f"Reason for Remedi Suggested: {row.get('Reason_for_Remedi_Suggested', 'N/A')}\n"
                        f"How to Use Remedi: {row.get('How_to_Use_Remedi', 'N/A')}\n"
                        f"Tips: {row.get('Tips', 'N/A')}\n"
                        "----------------------------------------\n"
                    )
                    print(output, end='')
                    f.write(output)
                    printed_symptoms.add(symptom)

# Run program
if __name__ == '__main__':
    symptoms_input = get_user_input()
    process_input(symptoms_input)
