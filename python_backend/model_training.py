import json
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import pandas as pd
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    ML_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML dependencies not available: {e}")
    print("To use model training, install dependencies:")
    print("pip install torch transformers scikit-learn pandas numpy matplotlib seaborn")
    ML_DEPENDENCIES_AVAILABLE = False

class BiasDataset(Dataset):
    def __init__(self, texts: List[str], labels: Dict[str, List[float]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        labels_tensor = {}
        for bias_type, values in self.labels.items():
            labels_tensor[bias_type] = torch.tensor(values[idx], dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels_tensor
        }

class BiasDetectionModel(nn.Module):
    """Multi-task bias detection model"""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_bias_types: int = 5, dropout: float = 0.3):
        super(BiasDetectionModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Individual heads for each bias type
        hidden_size = self.bert.config.hidden_size
        self.bias_heads = nn.ModuleDict({
            'gender': nn.Linear(hidden_size, 1),
            'racial': nn.Linear(hidden_size, 1),
            'political': nn.Linear(hidden_size, 1),
            'cultural': nn.Linear(hidden_size, 1),
            'overall': nn.Linear(hidden_size, 1)
        })
        
        # Shared feature layer
        self.feature_layer = nn.Linear(hidden_size, hidden_size // 2)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and feature transformation
        features = self.dropout(pooled_output)
        features = torch.relu(self.feature_layer(features))
        features = self.dropout(features)
        
        # Generate predictions for each bias type
        predictions = {}
        for bias_type, head in self.bias_heads.items():
            # Apply sigmoid to get scores between 0-1, then scale to 0-10
            predictions[bias_type] = torch.sigmoid(head(features)) * 10
        
        return predictions

class BiasModelTrainer:
    """Training pipeline for bias detection models"""
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.train_history = {'train_loss': [], 'val_loss': []}
        
    def load_dataset(self, dataset_path: str) -> Tuple[List[str], Dict[str, List[float]]]:
        """Load and preprocess training dataset"""
        print(f"📊 Loading dataset from {dataset_path}")
        
        # Load data (assuming JSON format)
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        elif dataset_path.endswith('.csv'):
            data = pd.read_csv(dataset_path).to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
        
        texts = []
        labels = {
            'gender': [],
            'racial': [],
            'political': [],
            'cultural': [],
            'overall': []
        }
        
        for item in data:
            texts.append(item['text'])
            
            # Extract labels (assuming they're in item['labels'])
            item_labels = item.get('labels', {})
            for bias_type in labels.keys():
                score = item_labels.get(f'{bias_type}_bias', 0.0)
                labels[bias_type].append(float(score))
        
        print(f"✅ Loaded {len(texts)} samples")
        return texts, labels
    
    def create_synthetic_dataset(self, num_samples: int = 1000) -> Tuple[List[str], Dict[str, List[float]]]:
        """Create synthetic dataset for demonstration purposes"""
        print(f"🎭 Creating synthetic dataset with {num_samples} samples")
        
        # Sample texts with different bias patterns
        sample_texts = [
            "The chairman announced the new policy to all employees.",
            "She is a great female engineer working in technology.",
            "The diverse team included members from various ethnic backgrounds.",
            "Conservative voters tend to support traditional values.",
            "The Muslim community celebrated their religious holiday.",
            "All people deserve equal treatment regardless of race.",
            "The spokesperson, a woman, addressed the media.",
            "Liberal policies often focus on social justice issues.",
            "Cultural differences should be respected and celebrated.",
            "The best candidate should be hired regardless of gender."
        ]
        
        texts = []
        labels = {
            'gender': [],
            'racial': [],
            'political': [],
            'cultural': [],
            'overall': []
        }
        
        # Generate samples with random variations
        for i in range(num_samples):
            # Pick a random base text and add variation
            base_text = np.random.choice(sample_texts)
            texts.append(base_text)
            
            # Generate synthetic bias scores
            gender_score = np.random.uniform(0, 8) if 'he' in base_text.lower() or 'she' in base_text.lower() or 'gender' in base_text.lower() else np.random.uniform(0, 3)
            racial_score = np.random.uniform(0, 7) if 'ethnic' in base_text.lower() or 'race' in base_text.lower() else np.random.uniform(0, 2)
            political_score = np.random.uniform(0, 9) if 'conservative' in base_text.lower() or 'liberal' in base_text.lower() else np.random.uniform(0, 3)
            cultural_score = np.random.uniform(0, 6) if 'cultural' in base_text.lower() or 'muslim' in base_text.lower() else np.random.uniform(0, 2)
            overall_score = np.mean([gender_score, racial_score, political_score, cultural_score])
            
            labels['gender'].append(gender_score)
            labels['racial'].append(racial_score)
            labels['political'].append(political_score)
            labels['cultural'].append(cultural_score)
            labels['overall'].append(overall_score)
        
        print(f"✅ Created synthetic dataset with {len(texts)} samples")
        return texts, labels
    
    def prepare_data(self, texts: List[str], labels: Dict[str, List[float]], 
                    test_size: float = 0.2, val_size: float = 0.1, batch_size: int = 16):
        """Prepare data loaders for training"""
        print("🔄 Preparing data loaders...")
        
        # Split data
        train_texts, test_texts, train_labels_dict, test_labels_dict = train_test_split(
            texts, 
            {k: v for k, v in labels.items()}, 
            test_size=test_size, 
            random_state=42
        )
        
        train_texts, val_texts, train_labels_dict, val_labels_dict = train_test_split(
            train_texts, 
            train_labels_dict, 
            test_size=val_size/(1-test_size), 
            random_state=42
        )
        
        # Create datasets
        train_dataset = BiasDataset(train_texts, train_labels_dict, self.tokenizer)
        val_dataset = BiasDataset(val_texts, val_labels_dict, self.tokenizer)
        test_dataset = BiasDataset(test_texts, test_labels_dict, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ Data prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, epochs: int = 10, learning_rate: float = 2e-5):
        """Train the bias detection model"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Initialize model
        self.model = BiasDetectionModel(self.model_name)
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(input_ids, attention_mask)
                
                # Calculate loss
                total_loss = 0
                for bias_type in predictions:
                    loss = criterion(predictions[bias_type].squeeze(), labels[bias_type])
                    total_loss += loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_train_loss += total_loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
            
            # Validation
            val_loss = self._validate_model(val_loader, criterion)
            
            # Record history
            avg_train_loss = total_train_loss / len(train_loader)
            self.train_history['train_loss'].append(avg_train_loss)
            self.train_history['val_loss'].append(val_loss)
            
            print(f"  📊 Training Loss: {avg_train_loss:.4f}")
            print(f"  📊 Validation Loss: {val_loss:.4f}")
            
        print("✅ Training completed!")
    
    def _validate_model(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                predictions = self.model(input_ids, attention_mask)
                
                total_loss = 0
                for bias_type in predictions:
                    loss = criterion(predictions[bias_type].squeeze(), labels[bias_type])
                    total_loss += loss
                
                total_val_loss += total_loss.item()
        
        return total_val_loss / len(val_loader)
    
    def evaluate_model(self, test_loader):
        """Evaluate the trained model"""
        print("📈 Evaluating model performance...")
        
        self.model.eval()
        all_predictions = {bias_type: [] for bias_type in ['gender', 'racial', 'political', 'cultural', 'overall']}
        all_labels = {bias_type: [] for bias_type in ['gender', 'racial', 'political', 'cultural', 'overall']}
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                predictions = self.model(input_ids, attention_mask)
                
                for bias_type in predictions:
                    all_predictions[bias_type].extend(predictions[bias_type].squeeze().cpu().numpy())
                    all_labels[bias_type].extend(labels[bias_type].cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        for bias_type in all_predictions:
            preds = np.array(all_predictions[bias_type])
            labels = np.array(all_labels[bias_type])
            
            metrics[bias_type] = {
                'mse': mean_squared_error(labels, preds),
                'mae': mean_absolute_error(labels, preds),
                'r2': r2_score(labels, preds)
            }
            
            print(f"  {bias_type.capitalize()} Bias:")
            print(f"    MSE: {metrics[bias_type]['mse']:.4f}")
            print(f"    MAE: {metrics[bias_type]['mae']:.4f}")
            print(f"    R²:  {metrics[bias_type]['r2']:.4f}")
        
        return metrics
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        print(f"💾 Saving model to {save_path}")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state and tokenizer
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'train_history': self.train_history
        }, save_path)
        
        # Save tokenizer
        tokenizer_path = save_path.replace('.pth', '_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
        
        print("✅ Model saved successfully!")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        print(f"📂 Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = BiasDetectionModel(checkpoint['model_name'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.train_history = checkpoint.get('train_history', {})
        
        # Load tokenizer
        tokenizer_path = model_path.replace('.pth', '_tokenizer')
        if os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        print("✅ Model loaded successfully!")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_history['train_loss'], label='Training Loss')
        plt.plot(self.train_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"📊 Training history plot saved to {save_path}")
        else:
            plt.show()

def main():
    """Main training script"""
    print("🔬 BiasGuard Model Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = BiasModelTrainer()
    
    # Create synthetic dataset for demonstration
    texts, labels = trainer.create_synthetic_dataset(num_samples=2000)
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(texts, labels, batch_size=8)
    
    # Train model
    trainer.train_model(train_loader, val_loader, epochs=5, learning_rate=2e-5)
    
    # Evaluate model
    metrics = trainer.evaluate_model(test_loader)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/bias_detector_{timestamp}.pth"
    trainer.save_model(model_path)
    
    # Plot training history
    trainer.plot_training_history(f"models/training_history_{timestamp}.png")
    
    print("🎉 Training pipeline completed successfully!")

if __name__ == "__main__":
    main()