from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('liuhaotian/llava-v1.5-7b')

# Prune the top 6 layers (adjust the number based on your needs)
model.transformer.h = model.transformer.h[:18]  # Keep the first 18 layers if it originally has 24

# Update the number of layers in the config
model.config.num_hidden_layers = len(model.transformer.h)