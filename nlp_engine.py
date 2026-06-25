import numpy as np
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class SentimentAnalyzer:
    def __init__(self):
        """Initializes the Hugging Face Pipeline inside our virtual environment."""
        print("🤖 Initializing Sentiment NLP Engine...")
        if pipeline is not None:
            # Using a lightweight, highly accurate DistilBERT variant (under 250MB)
            # It loads directly from Hugging Face into memory automatically on first run
            self.classifier = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.enabled = True
            print("✅ Real NLP Model loaded successfully via Hugging Face!")
        else:
            self.enabled = False
            print("⚠️ Hugging Face 'transformers' not installed. Running in mock backup mode.")

    def analyze_review(self, review_text: str):
        """
        Processes a textual review string and outputs structured data:
        Sentiment Label, Confidence Score, and Impact Array for the User Vector.
        """
        if self.enabled:
            # Run inference through the transformer model
            result = self.classifier(review_text)[0]
            label = result['label']       # 'POSITIVE' or 'NEGATIVE'
            confidence = result['score']  # Probability value between 0 and 1
        else:
            # Fallback mock logic if dependencies are missing during initial setup
            if "bad" in review_text.lower() or "fell apart" in review_text.lower():
                label, confidence = "NEGATIVE", 0.85
            else:
                label, confidence = "POSITIVE", 0.90

        # Calculate a profile impact vector shift based on the sentiment outcome
        # If Positive, it boosts interest; if Negative, it suppresses it.
        multiplier = 1.0 if label == "POSITIVE" else -1.0
        
        return {
            "text": review_text,
            "sentiment": label,
            "confidence": round(float(confidence), 4),
            "vector_impact_multiplier": multiplier
        }

if __name__ == "__main__":
    # Test our pipeline locally
    analyzer = SentimentAnalyzer()
    
    review_1 = "The acting by the lead was phenomenal, absolute masterpiece!"
    review_2 = "The third act plot completely fell apart and it was incredibly boring."
    
    print("\n--- Test 1 ---")
    print(analyzer.analyze_review(review_1))
    
    print("\n--- Test 2 ---")
    print(analyzer.analyze_review(review_2))