from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"

# Cargar tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cargar modelo
model = AutoModel.from_pretrained(model_name)

texto = "Artificial intelligence is fascinating"

# Tokenización
inputs = tokenizer(texto, return_tensors="pt")

print("Token IDs:")
print(inputs["input_ids"])

# Embeddings contextuales
outputs = model(**inputs)

print("Dimensiones de la representación:")
print(outputs.last_hidden_state.shape)
