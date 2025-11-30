from transformers import GPT2Tokenizer
from typing import List


class Tokenizer(object):
    """GPT-2 BPE tokenizer wrapper."""

    def __init__(self, vocab=None):
        """Initialize GPT-2 tokenizer. vocab parameter ignored for compatibility."""
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # GPT-2 uses <|endoftext|> for all special tokens
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, text: str) -> List[str]:
        """Encode text to list of token strings."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Convert IDs back to token strings for compatibility with corpus format
        return [self.tokenizer.decode([tid]) for tid in token_ids]

    def decode(self, tokens: List[str]) -> str:
        """Decode list of token strings to text."""
        # Join tokens and let GPT-2 tokenizer handle decoding
        text = ''.join(tokens)
        return text
