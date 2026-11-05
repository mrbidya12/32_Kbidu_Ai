# tree_of_thought.py
import torch
import torch.nn.functional as F
from collections import deque

class TreeOfThoughts:
    def __init__(self, model, tokenizer, device, max_steps=5, num_branches=3, width=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_steps = max_steps
        self.num_branches = num_branches
        self.width = width

    def generate(self, query, temperature=0.7):
        """Explore multiple reasoning paths and return the best."""
        prompt = f"Question: {query}\nLet's think step by step.\n"
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)

        # BFS search tree
        candidates = [(input_ids, 0.0)]  # (sequence, score)
        
        for step in range(self.max_steps):
            new_candidates = []
            for seq, score in candidates:
                if seq.size(1) > 512:  # avoid OOM
                    continue
                # Generate top-k next tokens
                with torch.no_grad():
                    logits = self.model(seq)
                    next_logits = logits[0, -1, :] / temperature
                    probs = F.softmax(next_logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, self.num_branches)
                
                for i in range(self.num_branches):
                    new_seq = torch.cat([seq, topk_indices[i].unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + torch.log(topk_probs[i] + 1e-10).item()
                    new_candidates.append((new_seq, new_score))
            
            # Keep only top 'width' candidates
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:self.width]
            
            # Stop if any candidate contains EOS
            for seq, _ in candidates:
                if seq[0, -1].item() == self.tokenizer.eos_id:
                    return self.tokenizer.decode(seq[0].tolist())

        # Return best sequence
        best_seq, _ = candidates[0]
        response = self.tokenizer.decode(best_seq[0].tolist())
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[-1].strip()
        return response