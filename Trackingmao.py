import cv2
import mediapipe as mp
import math


class DetectorMao:
    def __init__(self, modo=False, maxHands=2, confiancaDeteccao=0.5, confiancaRastreamento=0.5):
        self.modo = modo
        self.maxHands = maxHands
        self.confiancaDeteccao = confiancaDeteccao
        self.confiancaRastreamento = confiancaRastreamento

        self.mpHands = mp.solutions.hands
        self.maos = self.mpHands.Hands(static_image_mode=self.modo, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.confiancaDeteccao,
                                        min_tracking_confidence=self.confiancaRastreamento)
        self.mpDesenho = mp.solutions.drawing_utils
        self.dedoPontas = [4, 8, 12, 16, 20]
        self.dedos = []
        self.listaPontos = []

    def encontraMao(self, img, desenhar=True, flipTipo=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.resultados = self.maos.process(imgRGB)
        todasMaos = []
        altura, largura, canais = img.shape
        if self.resultados.multi_hand_landmarks:
            for tipoMao, pontosMao in zip(self.resultados.multi_handedness, self.resultados.multi_hand_landmarks):
                mao = {}
                ## listaPontos
                listaMao = []
                listaX = []
                listaY = []
                for id, ponto in enumerate(pontosMao.landmark):
                    px, py, pz = int(ponto.x * largura), int(ponto.y * altura), int(ponto.z * largura)
                    listaMao.append([px, py, pz])
                    listaX.append(px)
                    listaY.append(py)

                ## bbox
                xmin, xmax = min(listaX), max(listaX)
                ymin, ymax = min(listaY), max(listaY)
                larguraBox, alturaBox = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, larguraBox, alturaBox
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                mao["listaPontos"] = listaMao
                mao["bbox"] = bbox
                mao["centro"] = (cx, cy)

                if flipTipo:
                    if tipoMao.classification[0].label == "Right":
                        mao["tipo"] = "Left"
                    else:
                        mao["tipo"] = "Right"
                else:
                    mao["tipo"] = tipoMao.classification[0].label
                todasMaos.append(mao)

                ## desenhar
                if desenhar:
                    self.mpDesenho.draw_landmarks(img, pontosMao,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, mao["tipo"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if desenhar:
            return todasMaos, img
        else:
            return todasMaos

    def dedosLevantados(self, mao):
        tipoMao = mao["tipo"]
        listaMao = mao["listaPontos"]
        if self.resultados.multi_hand_landmarks:
            dedos = []
            if tipoMao == "Right":
                if listaMao[self.dedoPontas[0]][0] > listaMao[self.dedoPontas[0] - 1][0]:
                    dedos.append(1)
                else:
                    dedos.append(0)
            else:
                if listaMao[self.dedoPontas[0]][0] < listaMao[self.dedoPontas[0] - 1][0]:
                    dedos.append(1)
                else:
                    dedos.append(0)

            # 4 dedos
            for id in range(1, 5):
                if listaMao[self.dedoPontas[id]][1] < listaMao[self.dedoPontas[id] - 2][1]:
                    dedos.append(1)
                else:
                    dedos.append(0)
        return dedos
