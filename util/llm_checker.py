import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Carrega o modelo Huginn-0125 com bfloat16
model = AutoModelForCausalLM.from_pretrained(
    "tomg-group-umd/huginn-0125",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

# Se estiver usando GPU, certifique-se de mover os inputs para ela:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Cria uma configuração de geração personalizada
gen_config = GenerationConfig(
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    max_new_tokens=100
)

# Define um prompt de entrada
prompt = "Olá, mundo! Hoje vamos falar sobre inteligência artificial."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Gera a resposta
output_ids = model.generate(input_ids, generation_config=gen_config)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Texto gerado:")
print(output_text)
