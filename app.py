from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer,AutoModelForSequenceClassification,BertModel,BertTokenizer
from  backend import SentimentAnalyserLLM,SentimentAnalyserTransformer
import torch
import torch.nn as nn


app = FastAPI()

app.add_middleware(
    CORSMiddleware  ,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

llm = SentimentAnalyserLLM()
transformer = SentimentAnalyserTransformer()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class InputText(BaseModel):
    text: str

class BertSentimentModel(nn.Module):
  def __init__(self,model_name='bert-base-uncased',num_labels=3):
    super(BertSentimentModel,self).__init__()
    self.bert = BertModel.from_pretrained(model_name)
    self.dropout = nn.Dropout(0.3)
    self.fc = nn.Linear(self.bert.config.hidden_size,num_labels)

  def forward(self,input_ids,attention_mask):
    outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
    return self.fc(self.dropout(outputs.pooler_output))
  

llm_tokenizer = AutoTokenizer.from_pretrained('./sentiment_analysis_using_llm/saved_model/gpt_sentiment_tokenizer')
llm_model = AutoModelForSequenceClassification.from_pretrained('./sentiment_analysis_using_llm/saved_model/gpt_sent_trainer',num_labels=3).to(device)

transformer_tokenizer = BertTokenizer.from_pretrained('./sentiment_analysis_using_transformer/bert-tokenizer')
transformer_model = torch.load('./sentiment_analysis_using_transformer/saved_model/bert_sentiment_model.pth',map_location=torch.device(device),weights_only=False)

@app.get("/")  # Changed from @app.get to @app.post
async def main():
    return {"text":"Hello this is main page if you are here for sentiment analysis go to http://localhost:8000/predict page","device":device}

@app.post('/predict')
async def predict(input_data:InputText):
    text = [input_data.text]

    transformer_report = transformer.evaluate_example_texts(tokenizer=transformer_tokenizer,model=transformer_model,texts=text)
    llm_report = llm.evaluate_example_texts(tokenizer=llm_tokenizer,model=llm_model,texts=text)
    
    return {
        "transformer_report":{
        "text":transformer_report['text'],
        "sentiment":transformer_report['sentiment'],
        "confidence":transformer_report['confidence']
        },
        "llm_report":{
        "text":llm_report['text'],
        "sentiment":llm_report['sentiment'],
        "confidence":llm_report['confidence']
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)