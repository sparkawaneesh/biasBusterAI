# ML Model Training Guide for Bias Detection

## Overview
This guide provides comprehensive instructions for training custom machine learning models for bias detection, including dataset preparation, model training, and integration with the BiasGuard application.

## Dataset Requirements

### 1. Data Collection
- **Text Sources**: Articles, blog posts, social media content, news articles, academic papers
- **Bias Categories**: Gender, racial, political, cultural, age, religious, socioeconomic
- **Annotation**: Human-labeled bias scores (0-10 scale) for each category
- **Volume**: Minimum 50,000 samples for effective training

### 2. Dataset Structure
```
bias_detection_dataset/
├── train/
│   ├── gender_bias/
│   ├── racial_bias/
│   ├── political_bias/
│   ├── cultural_bias/
│   └── neutral/
├── validation/
└── test/
```

### 3. Data Format
```json
{
  "id": "unique_id",
  "text": "Article or content text",
  "labels": {
    "gender_bias": 3.5,
    "racial_bias": 1.2,
    "political_bias": 7.8,
    "cultural_bias": 2.1,
    "overall_bias": 4.2
  },
  "metadata": {
    "source": "news_article",
    "domain": "politics",
    "word_count": 450,
    "author": "anonymous"
  }
}
```

## Model Architecture Options

### 1. Transformer-Based Models

#### BERT-based Approach
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn

class BiasDetectionModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_bias_types=5):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1
        )
        self.bias_heads = nn.ModuleDict({
            'gender': nn.Linear(768, 1),
            'racial': nn.Linear(768, 1),
            'political': nn.Linear(768, 1),
            'cultural': nn.Linear(768, 1),
            'overall': nn.Linear(768, 1)
        })
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        bias_scores = {}
        for bias_type, head in self.bias_heads.items():
            bias_scores[bias_type] = torch.sigmoid(head(pooled_output)) * 10
            
        return bias_scores
```

#### RoBERTa-based Approach
```python
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn

class RoBERTaBiasDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 5)  # 5 bias types
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        return self.classifier(pooled_output)
```

### 2. Custom Neural Network Architecture

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class CustomBiasDetector(Model):
    def __init__(self, vocab_size, embedding_dim=128, max_length=512):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)
        self.lstm2 = layers.LSTM(64, dropout=0.2)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.bias_outputs = {
            'gender': layers.Dense(1, activation='sigmoid', name='gender_bias'),
            'racial': layers.Dense(1, activation='sigmoid', name='racial_bias'),
            'political': layers.Dense(1, activation='sigmoid', name='political_bias'),
            'cultural': layers.Dense(1, activation='sigmoid', name='cultural_bias'),
            'overall': layers.Dense(1, activation='sigmoid', name='overall_bias')
        }
        
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dropout(x)
        
        outputs = {}
        for bias_type, output_layer in self.bias_outputs.items():
            outputs[bias_type] = output_layer(x) * 10  # Scale to 0-10
            
        return outputs
```

## Training Pipeline

### 1. Data Preprocessing
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def preprocess_data(dataset_path):
    # Load dataset
    df = pd.read_json(dataset_path)
    
    # Text cleaning
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
    
    tokenized_data = df.apply(lambda x: tokenize_function({'text': x['text']}), axis=1)
    
    return tokenized_data, df

def create_data_loaders(tokenized_data, labels, batch_size=16):
    from torch.utils.data import DataLoader, TensorDataset
    
    dataset = TensorDataset(
        tokenized_data['input_ids'],
        tokenized_data['attention_mask'],
        torch.tensor(labels)
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### 2. Training Loop
```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss for each bias type
            loss = 0
            for bias_type in outputs:
                loss += criterion(outputs[bias_type], labels[bias_type])
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("-" * 50)

def validate_model(model, val_loader, criterion, device):
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask)
            
            loss = 0
            for bias_type in outputs:
                loss += criterion(outputs[bias_type], labels[bias_type])
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
```

### 3. Model Evaluation
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = {bias_type: [] for bias_type in ['gender', 'racial', 'political', 'cultural', 'overall']}
    actuals = {bias_type: [] for bias_type in ['gender', 'racial', 'political', 'cultural', 'overall']}
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids, attention_mask)
            
            for bias_type in outputs:
                predictions[bias_type].extend(outputs[bias_type].cpu().numpy())
                actuals[bias_type].extend(labels[bias_type].cpu().numpy())
    
    metrics = {}
    for bias_type in predictions:
        metrics[bias_type] = {
            'mse': mean_squared_error(actuals[bias_type], predictions[bias_type]),
            'mae': mean_absolute_error(actuals[bias_type], predictions[bias_type]),
            'r2': r2_score(actuals[bias_type], predictions[bias_type])
        }
    
    return metrics
```

## Integration with BiasGuard

### 1. Model Serving
```python
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer

app = Flask(__name__)

# Load trained model
model = BiasDetectionModel()
model.load_state_dict(torch.load('path/to/trained_model.pth'))
model.eval()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

@app.route('/predict', methods=['POST'])
def predict_bias():
    data = request.json
    text = data['text']
    
    # Tokenize input
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    # Convert to scores
    scores = {}
    for bias_type, score in outputs.items():
        scores[bias_type] = float(score.item())
    
    return jsonify(scores)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

### 2. Integration with BiasGuard Backend
```typescript
// server/services/custom-model.ts
import axios from 'axios';

export class CustomModelAnalyzer {
  private modelEndpoint: string;
  
  constructor(endpoint: string = 'http://localhost:5001') {
    this.modelEndpoint = endpoint;
  }
  
  async analyzeBias(text: string): Promise<BiasAnalysisResult> {
    try {
      const response = await axios.post(`${this.modelEndpoint}/predict`, {
        text: text
      });
      
      const scores = response.data;
      
      return {
        overallScore: scores.overall,
        genderScore: scores.gender,
        racialScore: scores.racial,
        politicalScore: scores.political,
        culturalScore: scores.cultural,
        detailedReport: {
          modelType: 'custom',
          confidence: scores.confidence || 0.85,
          sections: this.generateSections(scores)
        },
        wordCount: text.split(' ').length
      };
    } catch (error) {
      console.error('Custom model analysis failed:', error);
      throw new Error('Custom model analysis failed');
    }
  }
  
  private generateSections(scores: any) {
    const sections = [];
    
    for (const [biasType, score] of Object.entries(scores)) {
      if (biasType !== 'overall' && score > 3) {
        sections.push({
          type: biasType,
          riskLevel: score > 7 ? 'high' : score > 4 ? 'medium' : 'low',
          score: score,
          issues: [`Detected ${biasType} bias with score ${score}`],
          recommendations: [`Review content for ${biasType} bias patterns`]
        });
      }
    }
    
    return sections;
  }
}
```

## Production Deployment

### 1. Docker Configuration
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "model_server.py"]
```

### 2. Model Monitoring
```python
import logging
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Metrics
prediction_counter = Counter('bias_predictions_total', 'Total predictions made')
prediction_latency = Histogram('bias_prediction_duration_seconds', 'Prediction latency')
model_accuracy = Gauge('bias_model_accuracy', 'Model accuracy')

def monitor_prediction(func):
    def wrapper(*args, **kwargs):
        prediction_counter.inc()
        with prediction_latency.time():
            result = func(*args, **kwargs)
        return result
    return wrapper

@monitor_prediction
def predict_with_monitoring(text):
    # Your prediction logic here
    pass
```

## Best Practices

1. **Data Quality**: Ensure diverse, high-quality training data
2. **Bias Mitigation**: Implement fairness constraints during training
3. **Continuous Learning**: Regularly retrain with new data
4. **A/B Testing**: Compare custom model performance with GROQ
5. **Monitoring**: Track model performance and drift in production
6. **Privacy**: Implement data protection measures
7. **Scalability**: Design for horizontal scaling
8. **Versioning**: Maintain model version control

## Performance Optimization

1. **Model Quantization**: Reduce model size for faster inference
2. **Caching**: Cache frequent predictions
3. **Batch Processing**: Process multiple texts simultaneously
4. **GPU Acceleration**: Use CUDA for faster training and inference
5. **Model Pruning**: Remove unnecessary parameters

This guide provides a comprehensive foundation for developing and deploying custom bias detection models. Adapt the code and architecture based on your specific requirements and constraints.