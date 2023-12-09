# %%
from llama_cpp import Llama, llama_get_logits
import torch

#%%

llm = Llama(model_path="/home/ubuntu/csc2508/models/llama-2/llama-2-13b.Q5_K_M.gguf", n_gpu_layers=-1, logits_all=True) # , n_ctx=0)

# %%

llm.reset()
output = llm(
      "Q: What is the largest muscle in the body? A:", # Prompt
      # max_tokens=32, # Generate up to 32 tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=False,
      temperature=0,
      logprobs=1,
) # Generate a completion, can also call create_completion


print(output)
print(output["choices"][0]["logprobs"]["top_logprobs"])
#%%


# True token: 
true_token = llm.tokenize(bytes("True", "utf-8"))[1]
# False token
false_token = llm.tokenize(bytes("False", "utf-8"))[1]
#%%
# what I need for re-ranking:

llm.reset()
tokens = llm.tokenize(bytes("Q: True or False, The capital of the USA is Washington. A: True", "utf-8"))
llm.eval(tokens)
probs = torch.tensor(llm.scores).softmax(dim=1)
prob_true = probs[len(tokens)-2, true_token]
prob_false = probs[len(tokens)-2, false_token]
print(prob_true, prob_false)
if prob_true > prob_false:
    print("True")
else:
    print("False")
