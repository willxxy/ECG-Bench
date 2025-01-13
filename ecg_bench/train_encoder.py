import argparse

def get_args():
    parser = argparse.ArgumentParser(description = None)
    
    ### Data
    parser.add_argument('--data', type = str, default = None, help = 'Please choose the dataset')
    parser.add_argument('--seg_len', type = int, default = 500, help = 'Please choose the segment length')
    
    ### Model
    parser.add_argument('--model', type = str, default = None, help='Please choose the model')
    parser.add_argument('--device', type = str, default = None, help='Please choose the device')
    parser.add_argument('--seed', type = int, default = 0, help='Please choose the seed')
    parser.add_argument('--pad_to_max', type = int, default = 1000, help = 'Please specify the pad to max size')
    parser.add_argument('--tokenizer', type = str, help = 'Please specify the tokenizer')
    parser.add_argument('--percentiles', type = str, default = None, help = 'Please choose the percentiles computed during preprocessing')
    parser.add_argument('--peft', action = 'store_true', default = None, help = 'Please choose whether to use PEFT or not')
    
    ### Optimizer
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--batch_size', type = int, default = 128, help='Please choose the batch size')
    parser.add_argument('--epochs', type = int, default = 150, help='Please choose the number of epochs')
    parser.add_argument('--beta1', type = float, default = 0.9, help='Please choose beta 1 for optimizer')
    parser.add_argument('--beta2', type = float, default = 0.99, help='Please choose beta 2 for optimizer')
    parser.add_argument('--eps', type = float, default = 1e-8, help='Please choose epsilon for optimizer')
    parser.add_argument('--warmup', type = int, default = 500, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--patience', type = int, default = 5, help='Please choose the patience')
    
    ### For development
    parser.add_argument('--dev', action = 'store_true', help = 'Please choose whether to use development mode or not')
    parser.add_argument('--log', action = 'store_true', help = 'Please choose whether to log or not')
    
    ### Distributed Training
    parser.add_argument('--dis', action = 'store_true', help = 'Please choose whether to distributed training or not')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU ids to use (e.g., "0,1,2")')
    parser.add_argument('--ports', type=str, default='12356', help='Comma-separated list of ports to use (e.g., "12355,12356,12357")')
    
    return parser.parse_args()
    
def main():
    pass

if __name__ == '__main__':
    main(get_args())