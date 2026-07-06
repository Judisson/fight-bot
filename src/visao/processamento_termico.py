import cv2
import numpy as np

def _aplicar_gray_world(img):
  """
  Calcula a media global e normaliza os canais de cor, eliminando
  dominancias de iluminacao do fundo.
  """
  b, g, r = cv2.split(img)
  m_b, m_g, m_r = np.mean(b), np.mean(g), np.mean(r)
  mean_all = (m_b + m_g + m_r) / 3.0
  b = np.clip(b * (mean_all / (m_b + 1e-5)), 0, 255).astype(np.uint8)
  g = np.clip(g * (mean_all / (m_g + 1e-5)), 0, 255).astype(np.uint8)
  r = np.clip(r * (mean_all / (m_r + 1e-5)), 0, 255).astype(np.uint8)
  return cv2.merge((b, g, r))


class ProcessadorVisaoTermica:
  """
  Processador stateful de visao que acopla isolamento termico HSV com
  rastreio cinematico temporal atraves de fluxo otico de Farneback.
  Gera mapas de calor em 2D.
  """
  
  def __init__(self, frame_size=(128, 128)):
    self.frame_size = frame_size
    self.gray_anterior = None
    self.mapa_calor_acumulado = np.zeros((frame_size[1], frame_size[0]), dtype=np.float32)
    self.fator_decaimento = 0.8
    self.peso_movimento = 0.6
    self.peso_termico = 0.4
    self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

  def reset(self):
    self.gray_anterior = None
    self.mapa_calor_acumulado.fill(0.0)

  def processar_frame(self, frame_bgr):
    if frame_bgr is None:
      return np.zeros((self.frame_size[1], self.frame_size[0]), dtype=np.float32)

    # 1. Downscale e Gray World
    # Cv2 usa (width, height) no resize
    if frame_bgr.shape[:2] != (self.frame_size[1], self.frame_size[0]):
      frame_red = cv2.resize(frame_bgr, self.frame_size, interpolation=cv2.INTER_AREA)
    else:
      frame_red = frame_bgr

    if len(frame_red.shape) == 3:
      frame_calibrado = _aplicar_gray_world(frame_red)
      gray = cv2.cvtColor(frame_calibrado, cv2.COLOR_BGR2GRAY)
      hsv = cv2.cvtColor(frame_calibrado, cv2.COLOR_BGR2HSV)
      hsv[:, :, 2] = self._clahe.apply(hsv[:, :, 2])
    else:
      # Fallback se a imagem ja for grayscale
      frame_calibrado = frame_red
      gray = frame_red
      hsv = cv2.cvtColor(frame_red, cv2.COLOR_GRAY2BGR)
      hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
      hsv[:, :, 2] = self._clahe.apply(hsv[:, :, 2])

    # 2. Mascara Termica (Tons Quentes)
    limite_1 = np.array([0, 75, 75])
    limite_2 = np.array([15, 255, 255])
    limite_3 = np.array([165, 75, 75])
    limite_4 = np.array([180, 255, 255])

    m1 = cv2.inRange(hsv, limite_1, limite_2)
    m2 = cv2.inRange(hsv, limite_3, limite_4)
    mascara_termica = cv2.bitwise_or(m1, m2)
    densidade_termica = mascara_termica.astype(np.float32) / 255.0

    # 3. Movimento via Fluxo Otico (Farneback) e Compensacao de Ego-Motion
    magnitude_movimento = np.zeros((self.frame_size[1], self.frame_size[0]), dtype=np.float32)
    if self.gray_anterior is not None:
      # Calculo do deslocamento global da camera
      shift, _ = cv2.phaseCorrelate(self.gray_anterior.astype(np.float32), gray.astype(np.float32))
      dx, dy = shift

      fluxo = cv2.calcOpticalFlowFarneback(
        self.gray_anterior, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
      )
      
      # Subtrai o movimento da camera do fluxo total
      fluxo[..., 0] -= dx
      fluxo[..., 1] -= dy

      mag, _ = cv2.cartToPolar(fluxo[..., 0], fluxo[..., 1])
      magnitude_movimento = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    self.gray_anterior = gray

    # 4. Fusao do Mapa de Calor Acumulado
    self.mapa_calor_acumulado *= self.fator_decaimento
    energia_instantanea = (self.peso_movimento * magnitude_movimento) + (self.peso_termico * densidade_termica)
    self.mapa_calor_acumulado += energia_instantanea

    return np.clip(self.mapa_calor_acumulado, 0.0, 1.0)
