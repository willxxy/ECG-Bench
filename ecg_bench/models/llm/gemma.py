


import torch.nn as nn

### LLM FOR ECG-Byte
class gemma(nn.Module):
    def __init__(self, llm, args):
        super(gemma, self).__init__()
        self.args = args
        self.llm = llm
        if self.args.interpret:
            self.output_attentions = True
        else:
            self.output_attentions = False
            
    def forward(self, batch):
        if self.args.train == 'end2end':
            out = self.llm(input_ids = batch['input_ids'].to(self.llm.device),
                    attention_mask = batch['attn_mask'].to(self.llm.device),
                    labels = batch['labels'].to(self.llm.device),
                    position_ids = batch['position_ids'].to(self.llm.device),
                    output_attentions = self.output_attentions # this causes OOM during training so set it as False
                    )
        elif self.args.train == 'second':
            out = self.llm(inputs_embeds = batch['inputs_embeds'].to(self.llm.device),
                    attention_mask = batch['attn_mask'].to(self.llm.device),
                    labels = batch['labels'].to(self.llm.device),
                    position_ids = batch['position_ids'].to(self.llm.device),
                    output_attentions = self.output_attentions # this causes OOM during training so set it as False
                    )
        return out
    
    def generate(self, batch, tokenizer):
        input_len = batch['input_ids'].shape[1]
        if self.args.inference == 'second':
            generated_ids = self.llm.generate(
                    input_ids=batch['input_ids'].to(self.llm.device),
                    attention_mask=batch['attn_mask'].to(self.llm.device),
                    inputs_embeds=batch['inputs_embeds'].to(self.llm.device),
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    use_cache=True,
                )
        elif self.args.inference == 'end2end':
            generated_ids = self.llm.generate(
                    input_ids=batch['input_ids'].to(self.llm.device),
                    attention_mask=batch['attn_mask'].to(self.llm.device),
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    use_cache=True,
                )
        decoded_text = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded_text 
    
    
    def get_llm_embeddings(self, input_ids):
        out = self.llm.get_input_embeddings()(input_ids.to(self.llm.device))
        return out