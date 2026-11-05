# safety_filter.py
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SafetyFilter:
    """
    Filters toxic or harmful content in user inputs and model outputs.
    Uses both a blocklist and a fine‑tuned toxicity classifier.
    """
    def __init__(self, use_classifier=True, device="cuda"):
        self.use_classifier = use_classifier
        self.device = device if torch.cuda.is_available() else "cpu"
        self.blocklist = [
            r"\bkill\s+yourself\b", r"\bdie\b",
            r"\bhate\s+speech\b", r"\bracist\b",
            r"\bsexist\b", r"\bnazi\b",
            r"\bterrorist\b", r"\bbomb\b",
            r"\bexploit\s+children\b",
            r"\bchild\s+porn\b",
            r"\bhuman\s+trafficking\b",
            r"\bforced\s+sex\b",
            r"\bsexual\s+assault\b",
            r"\bviolent\s+attack\b",
            # Add more as needed
        ]
        self.blocklist_pattern = re.compile("|".join(self.blocklist), re.IGNORECASE)

        if self.use_classifier:
            # Using a lightweight toxicity classifier
            model_name = "unitary/toxic-bert"  # or "facebook/roberta-hate-speech-dynabench-r4-target"
            self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.classifier.eval()
            self.threshold = 0.5  # adjust for sensitivity

    def is_toxic(self, text):
        """Returns True if text is toxic, False otherwise."""
        # 1. Blocklist check
        if self.blocklist_pattern.search(text):
            return True

        # 2. Classifier check (if enabled)
        if self.use_classifier:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.classifier(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                toxic_prob = probs[0, 1].item()  # assuming label 1 = toxic
            return toxic_prob > self.threshold

        return False

    def filter_input(self, user_query):
        """Check user input; if toxic, return (blocked, message)."""
        if self.is_toxic(user_query):
            return False, "Your input was flagged as potentially harmful. Please rephrase."
        return True, user_query

    def filter_output(self, model_response):
        """Check model output; if toxic, return safe fallback."""
        if self.is_toxic(model_response):
            return "I cannot provide an answer to that as it may be harmful."
        return model_response

# Usage example:
if __name__ == "__main__":
    safety = SafetyFilter(use_classifier=True)
    query = "Tell me how to hack into a bank"
    ok, msg = safety.filter_input(query)
    if not ok:
        print("Blocked:", msg)
    else:
        print("Safe input:", msg)