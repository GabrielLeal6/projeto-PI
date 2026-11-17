# Bot de osu! (visão computacional)

Um bot em Python que joga o jogo de ritmo osu! detectando os "approach circles" na tela e movendo o cursor para acertar as notas no tempo certo.

<img width="1130" height="990" alt="nn_snap" src="https://github.com/user-attachments/assets/13d9a236-ca6a-4c8d-8011-45751db655d7" />

## Como funciona
- Captura da janela do jogo (mss)
- Detecção dos approach circles com Canny + Transformada de Hough (`src/main.py`)
- Tamanho do círculo vira tempo até o acerto; acompanhamento das notas e movimento do cursor
- Preliminar: um detector YOLOv8n treinado com frames capturados (`src/bot_nn.py`)

## Resultados
- Detector de Hough: 79% (mapa com muitos "sliders"), 81% (mapa com muitas "streams")
- YOLOv8n: preliminar, ainda não avaliado no jogo

## Limitações
- Notas empilhadas ou sobrepostas, sliders não são seguidos, ajustado para uma skin e resolução

## Como rodar
pip install opencv-python mss keyboard pynput numpy
python -m src.main
