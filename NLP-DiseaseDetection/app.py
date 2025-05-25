import torch
import joblib
import streamlit as st
import re
from transformers import BertTokenizer, BertForSequenceClassification

class MedicalSymptomClassifier:
    def __init__(self, model_path):
        """Initialize the medical symptom classifier with the specified model."""
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
        
        # Load model architecture
        self.num_labels = 3  # Assuming 3 classes: Respiratory, ENT, Musculoskeletal
        self.model = BertForSequenceClassification.from_pretrained(
            "monologg/biobert_v1.1_pubmed", 
            num_labels=self.num_labels
        )
        
        # Load saved model weights with appropriate error handling
        try:
            # Try loading as a complete model
            self.model = joblib.load(model_path)
            st.success(f"Successfully loaded model from {model_path}")
        except:
            try:
                # Try loading as a state dict
                state_dict = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                st.success(f"Successfully loaded model weights from {model_path}")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                raise Exception(f"Failed to load model: {e}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Map class indices to category names
        self.class_mapping = {
            0: "Respiratory",
            1: "ENT",
            2: "Musculoskeletal"
        }
        
        # Define keywords for each category
        self.keywords = {
            "Respiratory": ["cough", "breath", "lung", "chest", "wheez", "pneumonia", "asthma", 
                           "respiratory", "bronchitis", "copd", "dyspnea", "oxygen", "airway"],
            "ENT": ["ear", "nose", "throat", "hearing", "sinus", "voice", "tonsil", "hoarse", 
                   "nasal", "sore throat", "ent", "pharyn", "laryn", "tinnitus"],
            "Musculoskeletal": ["muscle", "joint", "bone", "pain", "weakness", "arthritis", 
                               "sprain", "tendon", "ligament", "stiff", "back pain", "swelling"]
        }
    
    def count_keywords(self, text):
        """Count occurrences of keywords for each category in the text."""
        text = text.lower()
        counts = {category: 0 for category in self.keywords}
        
        for category, keyword_list in self.keywords.items():
            for keyword in keyword_list:
                # Count occurrences of each keyword
                count = len(re.findall(r'\b' + keyword + r'\w*\b', text))
                counts[category] += count
                
        return counts
    
    def keyword_match(self, text):
        """Determine category based on keyword matching."""
        counts = self.count_keywords(text)
        if max(counts.values()) == 0:
            return None, counts
        
        # Get category with most keyword matches
        max_category = max(counts, key=counts.get)
        return max_category, counts
    
    def predict_bert(self, symptoms_text):
        """Predict category using BERT model."""
        # Tokenize input
        inputs = self.tokenizer(
            symptoms_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Get predicted class
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item() * 100
        
        # Get category name from class index
        predicted_category = self.class_mapping.get(
            predicted_class_idx, 
            f"Unknown Category (Class {predicted_class_idx})"
        )
        
        return predicted_category, confidence
    
    def analyze_symptoms(self, symptoms_text):
        """Complete analysis of symptoms with both keyword matching and BERT."""
        # Get keyword match results
        keyword_category, keyword_counts = self.keyword_match(symptoms_text)
        
        # Get BERT prediction
        bert_category, bert_confidence = self.predict_bert(symptoms_text)
        
        # Determine confidence level based on keyword match
        if keyword_category:
            max_count = keyword_counts[keyword_category]
            if max_count >= 3:
                confidence_text = "High (keyword match)"
            elif max_count >= 1:
                confidence_text = "Medium (keyword match)"
            else:
                confidence_text = "Low"
        else:
            confidence_text = "Low"
            
        # Use keyword match if available, otherwise use BERT prediction
        final_category = keyword_category if keyword_category else bert_category
        
        # Create result dictionary
        result = {
            "symptoms": symptoms_text,
            "prediction": final_category,
            "confidence": confidence_text,
            "keyword_match": keyword_category,
            "keyword_counts": keyword_counts,
            "bert_prediction": bert_category,
            "bert_confidence": bert_confidence
        }
        
        return result


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Medical Symptom Classifier",
        page_icon="🏥",
        layout="centered"
    )
    
    # Title and header
    st.title("Medical Symptom Classifier")
    st.markdown("### Analyze symptoms by category")
    
    # Add disclaimer with warning styling
    st.warning("**DISCLAIMER**: This application is for educational purposes only. Always consult a healthcare professional for medical advice.")
    
    # Create sidebar with information
    with st.sidebar:
        st.header("About")
        st.info(
            "This application classifies medical symptoms into categories "
            "using both keyword matching and BERT model prediction."
        )
        st.markdown("### Categories")
        st.markdown(
            "- **Respiratory**: Breathing-related symptoms\n"
            "- **ENT**: Ear, Nose, and Throat symptoms\n"
            "- **Musculoskeletal**: Muscle, joint, and bone symptoms"
        )
    
    # Model path input (for flexibility)
    MODEL_PATH = st.sidebar.text_input(
        "Model Path",
        value="medical_symptom_classifier.joblib",
        help="Path to your trained model file"
    )
    
    # Check if the model is already loaded in session state
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
        st.session_state.model_path = None
    
    # Main content - Symptom input
    st.subheader("Describe Your Symptoms")
    symptom_text = st.text_area(
        "Please provide a description of the symptoms:",
        height=100,
        placeholder="Example: cough and difficulty breathing"
    )
    
    # Analyze button
    analyze_button = st.button("Analyze Symptoms", type="primary")
    
    # Display previous results
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Process when button is clicked and symptoms are provided
    if analyze_button:
        if not symptom_text.strip():
            st.error("Please enter symptoms before analyzing.")
        else:
            with st.spinner("Analyzing symptoms..."):
                try:
                    # Initialize model if not already loaded or if model path changed
                    if st.session_state.classifier is None or st.session_state.model_path != MODEL_PATH:
                        st.session_state.classifier = MedicalSymptomClassifier(MODEL_PATH)
                        st.session_state.model_path = MODEL_PATH
                    
                    # Get prediction
                    result = st.session_state.classifier.analyze_symptoms(symptom_text)
                    
                    # Add to results history
                    st.session_state.results.append(result)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.info("Please make sure the model file exists and is in the correct format.")
    
    # Always display results history if available
    if st.session_state.results:
        st.subheader("Analysis Results")
        
        for result in st.session_state.results:
            with st.container():
                st.markdown("---")
                st.markdown(f"""
                **Symptoms:** {result['symptoms']}
                
                **Prediction:** {result['prediction']} (Confidence: {result['confidence']})
                
                **Keyword Match:** {result['keyword_match'] or 'None'}
                
                **Keyword Counts:** {result['keyword_counts']}
                
                **BERT Prediction:** {result['bert_prediction']}
                """)
    
    # Clear results button
    if st.session_state.results:
        if st.button("Clear Results"):
            st.session_state.results = []
            st.experimental_rerun()


if __name__ == "__main__":
    main()