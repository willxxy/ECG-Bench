import torch

path = './runs/ecg-qa_ptbxl_mapped_1250/0/llama-3.2-1b-instruct_8_1_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None/best_model.pth'

model = torch.load(path)

print(model['epoch'])