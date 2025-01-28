import torch.nn as nn

class LLM(nn.Module):
    def __init__(self, llm, args):
        super(LLM, self).__init__()
        self.args = args
        self.llm = llm
        if self.args.interpret:
            self.output_attentions = True
        else:
            self.output_attentions = False
            
    def forward(self, batch):

        #         out = self.llm(input_ids = batch['tokenized_signal'].to(self.llm.device),
        #             attention_mask = batch['attn_mask'].to(self.llm.device),
        #             labels = batch['quantized_signal_ids_input'].to(self.llm.device),
        #             position_ids = batch['position_ids'].to(self.llm.device),
        #             output_attentions = self.output_attentions # this causes OOM during training so set it as False
        #             )
        # return out


        out = self.llm(input_ids = batch['tokenized_question'].to(self.llm.device),
                    attention_mask = batch['attention_mask'].to(self.llm.device),
                    labels = batch['tokenized_answer'].to(self.llm.device),
                    )
        return out
    
    def generate(self, batch, tokenizer):
        input_len = batch['tokenized_signal'].shape[1]
        generated_ids = self.llm.generate(
                input_ids=batch['tokenized_signal'].to(self.llm.device),
                attention_mask=batch['attn_mask'].to(self.llm.device),
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
            )
        decoded_text = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded_text 
    



# print('Initializing Model')
# tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir = './../.huggingface')
# llm = AutoModelForCausalLM.from_pretrained(args.model, cache_dir = './../.huggingface', torch_dtype=torch.bfloat16)

# new_ids = list(vocab.keys())
# new_ids = [f'signal_{str(ids)}' for ids in new_ids]
# tokenizer.add_tokens(new_ids)
# tokenizer.add_tokens(['<sig_start>'], special_tokens=True)
# tokenizer.add_tokens(['<sig_end>'], special_tokens=True)
# tokenizer.add_special_tokens({"pad_token":"<pad>"})
# llm.config.pad_token_id = tokenizer.pad_token_id
# llm.resize_token_embeddings(len(tokenizer))

# if args.peft:
#     llm = get_peft_model(llm, lora_config)
#     llm.print_trainable_parameters()
    
# model = LLM(llm, args)
# model = model.to(device)
# model_hidden_size = model.llm.config.hidden_size
# find_unused_parameters = False


    
    # print(f'Total number of parameters: {count_parameters(model)}')
    
    # if args.dis:
    #     model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)