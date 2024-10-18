import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the CSV data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Custom dataset for BERT
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load BERT Model
def load_model(model_name='bert-base-uncased', num_labels=2):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

# Fine-tuning process (example)
def train_model(model, tokenizer, train_data, val_data, epochs=3, batch_size=16):
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    accuracy = accuracy_score(actuals, predictions)
    return model, accuracy

# Streamlit app
def main():
    st.title("Text Classification with BERT")

    # Upload the data
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write(data.head())

        # Define target and features
        texts = data['text']  # Assume 'text' is the column name for input data
        labels = data['label']  # Assume 'label' is the column name for labels

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

        # Load model and tokenizer
        tokenizer, model = load_model()
        train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len=128)
        val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len=128)

        if st.button('Train Model'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            model, accuracy = train_model(model, tokenizer, train_dataset, val_dataset)
            st.write(f"Training Complete. Validation Accuracy: {accuracy:.2f}")

        # Text input for prediction
        st.write("### Test the model with your own input:")
        user_input = st.text_area("Enter text here", "")
        
        if user_input:
            encoded_input = tokenizer.encode_plus(
                user_input,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
            st.write(f"Prediction: {preds[0]}")  # You can map it back to the label names if necessary.

if __name__ == '__main__':
    main()
