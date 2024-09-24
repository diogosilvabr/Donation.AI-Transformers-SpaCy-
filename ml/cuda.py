import torch
print(torch.cuda.is_available())  # Deve retornar True se a GPU estiver ativa
print(torch.cuda.device_count())  # Verifica o número de GPUs disponíveis
print(torch.cuda.get_device_name(0))  # Exibe o nome da GPU em uso
