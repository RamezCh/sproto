from transformers import AutoModel, AutoConfig

model_path = "output/sproto_hf"

config = AutoConfig.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

print(type(config))
print(type(model))