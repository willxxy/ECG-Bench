import llm_blender
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM", device = 'cuda:7',cache_dir = './../.huggingface')

inputs = ["hello!"]
candidates_A = ["hi!"]
candidates_B = ["f**k off!"]
comparison_results = blender.compare(inputs, candidates_A, candidates_B)
print(comparison_results)