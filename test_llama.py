# %%
from llama_cpp import Llama, llama_get_logits
import torch

#%%

llm = Llama(model_path="/home/ubuntu/csc2508/models/llama-2/llama-2-13b.Q5_K_M.gguf", n_gpu_layers=-1, logits_all=True, n_ctx=2048) # , n_ctx=0)

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

# %%


caption = "a man in military dress and a woman in a white and dotted dress are kissing"
prompt = f"""The following is a transcript of the images in a video:
The image shows a man and a woman standing on a staircase, with the man holding the woman in his arms. They are both looking at each other, possibly sharing a tender moment or a romantic gesture. The woman is wearing a dress, and the man is wearing a tie, suggesting a formal or semi-formal occasion. The presence of a handbag nearby adds to the overall ambiance of the scene.
The image depicts and a woman standing on a staircase, with the woman leaning on the stomach. The woman is wearing a necklace around their necks. The staircase is located on the left side of the staircase is on the right. The scene suggests a formal or special occasion. The presence of the staircase and the woman's the overall atmosphere of the scene.
The image shows a man standing on a staircase, holding a woman in his arms. The man is wearing a uniform, which suggests that he might be a police officer or a security personnel. The woman appears to be in distress, and the man is trying to comfort her. The scene takes place in a location with a lot of water, possibly a flooded area or a beach. The presence of water adds an element of urgency to the situation, as the man and woman seem to be dealing with a challenging situation together.
In the image, a man and a woman are standing on a staircase, with the man holding a hat in his hand. The woman is wearing a dress and appears to be looking at the man. The scene takes place in a room with a potted plant nearby, and there is a bottle on the floor. The presence of the water and the staircase suggests that they might be in a location with a water source, such as a beach or a poolside. The man's hat could be a sun hat, indicating that they are in a sunny or warm environment.
In the image, a man and a woman are standing on a staircase, with the man holding a hat in his hand. The woman is wearing a white dress, and the man is wearing a brown uniform. The scene appears to be set in a location where there is a lot of water, possibly near a beach or a waterfront. The man is holding the hat in front of the woman, possibly to protect her from the wind or the sun. The woman is looking at the man, possibly engaging in a conversation or observing the surroundings.
In the image, a man and a woman are standing next to each other, with the man holding a hat in his hand. The woman is wearing a dress and appears to be looking at the man. The man is wearing a tie, which suggests that he might be dressed in formal attire. The presence of a tie and the fact that the man is holding a hat indicate that the scene might be taking place in a formal or semi-formal setting. The woman's gaze towards the man could imply that she is engaged in a conversation or paying attention to something the man is saying or doing.
The image shows a man and a woman standing next to each other, with the man holding a hat in his hand. They are standing in front of a body of water, possibly a lake or a river. The woman is holding a hat in her hand, and the man is wearing a tie. The scene appears to be a casual and relaxed, with the man and woman enjoying their time together near the water.
The image shows a man and a woman standing next to each other, with the man wearing a uniform and the woman wearing a floral print shirt. The man is holding a large umbrella, and the woman is holding a fan. They are standing in front of a body of water, possibly a lake or a river. The man appears to be pointing at something in the water, possibly drawing the woman's attention to it. The scene suggests that they are enjoying a leisurely outdoor activity together, possibly taking a break from their daily routine or engaging in a conversation.
The image shows a man and a woman standing next to each other, with the man wearing a uniform and the woman wearing a dress. The man is wearing a tie, which suggests that he might be a police officer. They are standing in front of a potted plant, and there is a handbag nearby. The man appears to be talking to the woman, possibly giving her directions or discussing something important. The scene seems to be taking place in a room with a chair and a clock on the wall.
The image shows a man and a woman standing next to each other in a room, engaged in a conversation. The man is wearing a brown shirt and tie, while the woman is wearing a white shirt. They are standing close to each other, with the man on the left and the woman on the right. The woman appears to be talking to the man, who is listening attentively.\n\nIn the background, there is a potted plant on the left side of the room, and a clock on the wall. The room seems to be a living room or a similar indoor setting.
The image features a man and a woman standing close to each other, with the man wearing a tie and the woman wearing a dress. The man is holding the woman's face in his hands, and they appear to be having a conversation. The woman is wearing a tie around her neck, which is an unusual accessory for a woman. The scene takes place in a room with a potted plant in the background, and there is a clock on the wall. The man is also wearing a tie, which is a common accessory for men's formal attire.
The image shows a man and a woman standing close to each other, with the man kissing each other's cheek. They are standing in front of a painting on the wall, which is hanging on the wall. The man is wearing a tie, and the woman is wearing a dress. They are both smiling and enjoying their time together.
Q: True or False, this is a good caption for the video: {caption}. A: True"""

llm.reset()
tokens = llm.tokenize(bytes(prompt, "utf-8"))
print(len(tokens))
llm.eval(tokens)
probs = torch.tensor(llm.scores).softmax(dim=1)
prob_true = probs[len(tokens)-2, true_token]
prob_false = probs[len(tokens)-2, false_token]
print(prob_true, prob_false)
if prob_true > prob_false:
    print("True")
else:
    print("False")