import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BillingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang):
        super().__init__()
        self.ds = ds  # The dataset containing the parallel corpora
        self.tokenizer_src = tokenizer_src  # Tokenizer for the source language
        self.tokenizer_tgt = tokenizer_tgt  # Tokenizer for the target language
        self.src_lang = src_lang  # Source language code
        self.tgt_lang = tgt_lang  # Target language code
        
        # Tokens for start-of-sequence, end-of-sequence, and padding
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]
        
        # Tokenize the source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # Create the encoder input without padding
        encoder_input = torch.tensor(
            [self.sos_token.item()] + enc_input_tokens + [self.eos_token.item()],
            dtype=torch.int64
        )
        
        # Create the decoder input without padding
        decoder_input = torch.tensor(
            [self.sos_token.item()] + dec_input_tokens,
            dtype=torch.int64
        )
        
        # Create the label without padding
        label = torch.tensor(
            dec_input_tokens + [self.eos_token.item()],
            dtype=torch.int64
        )
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.int)  # Upper triangular mask
    return mask == 0  # Convert to boolean mask
