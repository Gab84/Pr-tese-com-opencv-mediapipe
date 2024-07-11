import Trackingmao
import cv2

camera = cv2.VideoCapture(0)

detector = Trackingmao.DetectorMao(confiancaDeteccao=0.9, maxHands=2)

while True:
    _, img = camera.read()
    img = cv2.flip(img, 1)
    maos, img = detector.encontraMao(img)
    # sem as marcações basta tirar o img e colocar desenhar=False no encontraMao
    # print(len(maos))
    
    if maos:
        # mao 1
        mao1 = maos[0]  # dicionário mão 1
        pontosm1 = mao1['listaPontos']  # lista dos 21 pontos das mãos
        boxmao1 = mao1['bbox']  # Caixa que fica em contorno da mão
        pontocentral1 = mao1['centro']  # o ponto central da mão
        dedos = detector.dedosLevantados(mao1)
        print(dedos)
        
    # caso queira mais de uma mão
    # if len(maos) == 2:
    #     mao2 = maos[1]  # dicionário mão 2
    #     pontosm2 = mao2['listaPontos']  # lista dos 21 pontos das mãos
    #     boxmao2 = mao2['bbox']  # Caixa que fica em contorno da mão
    #     pontocentral2 = mao2['centro']  # o ponto central da mão
    
    cv2.imshow('Oioi', img)
    cv2.waitKey(1)
