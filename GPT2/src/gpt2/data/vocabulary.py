from transformers import GPT2Tokenizer
from typing import Union


class Vocab(object):
    """GPT-2 vocabulary adapter."""

    def __init__(self, vocab_path: str = None, **kwargs):
        """Initialize GPT-2 vocabulary. vocab_path ignored - uses pretrained."""
        self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

        # Cache the vocabulary for fast lookups (token string -> token ID)
        self._token_to_id = self.tokenizer.get_vocab()

        # GPT-2 uses <|endoftext|> for all special tokens
        self.unk_token = '<|endoftext|>'
        self.bos_token = '<|endoftext|>'
        self.eos_token = '<|endoftext|>'
        self.pad_token = '<|endoftext|>'
        self.additional_tokens = [self.bos_token, self.eos_token, self.pad_token]

    def __getitem__(self, idx_or_token: Union[int, str]) -> Union[str, int]:
        if isinstance(idx_or_token, str):
            # Token to ID - use cached vocabulary for O(1) lookup
            return self._token_to_id.get(idx_or_token, self.unk_idx)
        else:
            # ID to token
            return self.tokenizer.decode([idx_or_token])

    def __contains__(self, token: str) -> bool:
        return token in self.tokenizer.get_vocab()

    def __len__(self) -> int:
        # Pad to multiple of 8 like original
        vocab_size = len(self.tokenizer)  # 50,257
        return (vocab_size + 7) // 8 * 8  # Returns 50,264

    @property
    def unk_idx(self) -> int:
        return self.tokenizer.eos_token_id  # 50,256

    @property
    def bos_idx(self) -> int:
        return self.tokenizer.eos_token_id  # 50,256

    @property
    def eos_idx(self) -> int:
        return self.tokenizer.eos_token_id  # 50,256

    @property
    def pad_idx(self) -> int:
        return self.tokenizer.eos_token_id  # 50,256
