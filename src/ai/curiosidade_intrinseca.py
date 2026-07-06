import torch
import torch.nn as nn
import torch.nn.functional as F

class CodificadorVisualICM(nn.Module):
    """
    Codificador do estado visual para um espaço latente limpo.
    Focado em descartar ruídos do cenário (Problema da TV Barulhenta).
    """
    def __init__(self, canais_entrada, tamanho_frame, tamanho_latente=256):
        super().__init__()
        # CNN para extração espacial robusta
        self.conv = nn.Sequential(
            nn.Conv2d(canais_entrada, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calcula tamanho do flatten automaticamente com base nas dimensões
        with torch.no_grad():
            dummy = torch.zeros(1, canais_entrada, tamanho_frame, tamanho_frame)
            saida = self.conv(dummy)
            total_flat = int(saida.numel() / saida.shape[0])
            
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_flat, tamanho_latente),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class ModeloInversoDinamicas(nn.Module):
    """
    Recebe phi(s_t) e phi(s_{t+1}) e tenta prever a acao a_t que operou a transição.
    Isto treina o encoder a manter APENAS informação que muda sob as ações do agente.
    """
    def __init__(self, tamanho_latente, total_acoes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(tamanho_latente * 2, 256),
            nn.ReLU(),
            nn.Linear(256, total_acoes)
        )
        
    def forward(self, phi_st, phi_st_plus_1):
        # Concatena o estado latente atual e o próximo estado latente ao longo do eixo das features
        x = torch.cat([phi_st, phi_st_plus_1], dim=1)
        # Retorna os logits de predição da ação
        return self.fc(x)

class ModeloDiretoDinamicas(nn.Module):
    """
    Recebe phi(s_t) e a_t, e prevê \hat{\phi}(s_{t+1}).
    O erro desta predição será usado como sinal de curiosidade (recompensa intrínseca).
    """
    def __init__(self, tamanho_latente, total_acoes):
        super().__init__()
        self.total_acoes = total_acoes
        self.fc = nn.Sequential(
            nn.Linear(tamanho_latente + total_acoes, 256),
            nn.ReLU(),
            nn.Linear(256, tamanho_latente)
        )
        
    def forward(self, phi_st, acao):
        # A acao é esperada como um tensor de índices (batch_size,)
        # Convertemos para One-Hot Encoding
        acao_one_hot = F.one_hot(acao, num_classes=self.total_acoes).float()
        
        # Concatena estado latente e ação
        x = torch.cat([phi_st, acao_one_hot], dim=1)
        
        # Retorna predição do próximo estado latente \hat{\phi}(s_{t+1})
        return self.fc(x)
