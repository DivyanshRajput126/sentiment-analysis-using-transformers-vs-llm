from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer,BertModel
import torch
import torch.nn as nn
from torch.nn import functional

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BertSentimentModel(nn.Module):
  def __init__(self,model_name='bert-base-uncased',num_labels=3):
    super(BertSentimentModel,self).__init__()
    self.bert = BertModel.from_pretrained(model_name)
    self.dropout = nn.Dropout(0.3)
    self.fc = nn.Linear(self.bert.config.hidden_size,num_labels)

  def forward(self,input_ids,attention_mask):
    outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
    return self.fc(self.dropout(outputs.pooler_output))


class SentimentAnalyserLLM:
    # Evaluate Model with Example Texts
    def evaluate_example_texts(self,tokenizer,model,texts):
        model.eval()
        model.to(device)
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, truncation=True, return_tensors='pt')
                outputs = model(**inputs)

                probs = functional.softmax(outputs.logits,dim=-1)
                pred_label =  torch.argmax(probs,dim=1).item() # Placeholder logic for sentiment extraction from Llama output
                confidence = probs[0,pred_label].item()
                sentiment = label_map[pred_label]
                confidence = confidence*100
                confidence = "{:.4f}".format(confidence)
                confidence = float(confidence)

                print(f"Text: {text}\nPredicted Sentiment: {sentiment}, (Confidence = {confidence}) ({type(confidence)})\n")
                return({
                    "text":text,
                    'sentiment': sentiment,
                    # 'actual_sentiment':actual_sentiment,
                    'confidence':{confidence}
                })

# Evaluate model
class SentimentAnalyserTransformer:
  def evaluate_example_texts(self,tokenizer,model,texts):
    model.eval()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, truncation=True, return_tensors='pt')
            input_ids,attention_mask = inputs['input_ids'],inputs['attention_mask']
            outputs = model(input_ids,attention_mask)

            if isinstance(outputs,torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits

            probs = functional.softmax(logits,dim=-1)
            pred_label =  torch.argmax(probs,dim=1).item() # Placeholder logic for sentiment extraction from Llama output
            confidence = probs[0,pred_label].item()
            confidence = confidence*100
            confidence = "{:.4f}".format(confidence)
            confidence = float(confidence)
            sentiment = label_map[pred_label]

            print(f"Text: {text}\nPredicted Sentiment: {sentiment}, (Confidence = {confidence}) \n")
            return({
                "text":text,
                'sentiment': sentiment,
                # 'actual_sentiment':actual_sentiment,
                'confidence':{confidence}
            })