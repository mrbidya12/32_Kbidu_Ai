# inference.py
import torch
import torch.nn.functional as F
from collections import Counter
import re

class Inference:
    def __init__(self, model, tokenizer, device, retriever=None, max_new_tokens=300, memory_size=5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.retriever = retriever
        self.max_new_tokens = max_new_tokens
        self.memory = []
        self.memory_size = memory_size

    def _build_prompt(self, query, use_cot=True, tone="empathetic", think_before_answer=False):
        tone_map = {
            "empathetic": "You are Kbidu AI, a warm, kind, and emotionally intelligent assistant.",
            "professional": "You are Kbidu AI, a precise, clear, and objective assistant.",
            "humorous": "You are Kbidu AI, a witty and fun assistant.",
            "supportive": "You are Kbidu AI, a supportive and encouraging mentor."
        }
        system_prompt = tone_map.get(tone, tone_map["empathetic"])

        # RAG context
        context = ""
        if self.retriever:
            results = self.retriever.retrieve(query, top_k=3)
            if results:
                context = "Relevant information:\n" + "\n".join([r["chunk"] for r in results]) + "\n\n"

        # Conversation memory
        memory_text = ""
        if self.memory:
            memory_text = "Conversation history:\n"
            for user, assistant in self.memory[-self.memory_size:]:
                memory_text += f"User: {user}\nAssistant: {assistant}\n"
            memory_text += "\n"

        # Think-before-answer instruction
        think_instruction = ""
        if think_before_answer:
            think_instruction = (
                "Before giving your final answer, think step by step inside your head.\n"
                "Then, provide your final answer after the line 'Final Answer:'.\n"
                "Do not include your reasoning in the final output unless asked.\n"
            )

        cot_instruction = "Let's think step by step.\n" if use_cot else ""

        prompt = f"{system_prompt}\n\n{context}{memory_text}{think_instruction}{cot_instruction}User: {query}\nAssistant:"
        return prompt

    def generate(self, query, use_cot=True, temperature=0.7, tone="empathetic", think_before_answer=False, show_reasoning=False):
        prompt = self._build_prompt(query, use_cot, tone, think_before_answer)
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)

        for _ in range(self.max_new_tokens):
            with torch.no_grad():
                logits = self.model(input_ids)
                next_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_id:
                    break

        full_response = self.tokenizer.decode(input_ids[0].tolist())

        # Extract assistant part
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:", 1)[-1].strip()
        else:
            response = full_response[len(prompt):].strip()

        # If think_before_answer is True, try to extract final answer after "Final Answer:"
        if think_before_answer:
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:", 1)[-1].strip()
                if show_reasoning:
                    # Return the full response (reasoning + final answer)
                    self.memory.append((query, response))
                    return response
                else:
                    # Return only the final answer
                    self.memory.append((query, final_answer))
                    return final_answer
            else:
                # If no "Final Answer:" found, return the full response (fallback)
                self.memory.append((query, response))
                return response
        else:
            self.memory.append((query, response))
            return response

    def generate_with_self_consistency(self, query, n=5, temperature=0.7, tone="empathetic", think_before_answer=False):
        answers = []
        for _ in range(n):
            ans = self.generate(query, use_cot=True, temperature=temperature, tone=tone, think_before_answer=think_before_answer, show_reasoning=False)
            if "Answer:" in ans:
                ans = ans.split("Answer:")[-1].strip()
            answers.append(ans)
        if answers:
            counter = Counter(answers)
            return counter.most_common(1)[0][0]
        return answers[0] if answers else ""