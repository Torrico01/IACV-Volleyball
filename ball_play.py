import numpy as np
import cv2 as cv
import os
import ball_net as bn
import blobber
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
maxfev_test = 2000
################ Função quadrática para aproximar a trajetória
def func(x,a,b,c):
  return a*(x-b)**2+c
###############################################################

def draw_ball(mask, frame):
  cnts, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

  k = 0
  for c in cnts:
    rx,ry,rw,rh  = cv.boundingRect(c)
    mn = min(rw, rh)
    mx = max(rw, rh)
    r = mx / mn
    if mn < 10 or mx > 40 or r > 1.5:
      continue

    cut_m = mask[ry : ry + rh, rx : rx + rw]
    cut_f = frame[ry : ry + rh, rx : rx + rw]

    cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut_m)
    if bn.check_pic(cut_c) == 0:
      ((x, y), r) = cv.minEnclosingCircle(c)
      cv.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 3)


def test_clip(path):
  vs = cv.VideoCapture(path)
  backSub = cv.createBackgroundSubtractorMOG2()
  n = 0
  frames = []
  npframe = np.array([])

  ###### Inicialização de variáveis para calcular as trajetórias
  x_data = [] # Pontos x da trajetória
  y_data = [] # Pontos y da trajetória
  data_abs = [[]] # Lista de pontos [[[x1,y1],...,[xn,yn]],...] para cada trajetória detectada
  data_popt = [] # Lista de parâmetros [[a,b,c],...] do ajuste de curva para cada trajetória detectada
  path_counter = 0 # Conta quantas trajetórias foram detectadas
  delta_perr = 0.4 # Delta de quanto o erro do ajuste de curva deve subir para ser considerado uma nova trajetória
  perr_mean_ant = -delta_perr # Armazena o valor anterior do erro do ajuste de curva
  ###############################################################

  while(True):
    ret, frame = vs.read()
    if not ret or frame is None:
      break

    h = frame.shape[0]
    w = frame.shape[1]

    frame = cv.resize(frame, (int(w/2),int(h/2)))
    mask = backSub.apply(frame)
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    mask = cv.dilate(mask, None)
    mask = cv.GaussianBlur(mask, (15, 15),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)

    prev_bb = blobber.handle_blobs(mask, frame)

    ############## Detecção de trajetória
    x_data = []
    y_data = []

    if prev_bb != None and len(prev_bb.pts)>8:
      for point in prev_bb.pts:
        if not point in data_abs[path_counter]:
          x_data.append(point[0])
          y_data.append(point[1])

      if len(x_data)>2:
        popt, pcov = curve_fit(func, x_data, y_data, maxfev = maxfev_test)
        perr = np.sqrt(np.diag(pcov))
        perr_mean = np.mean(perr)

        '''
        npframe = np.array(frame)
        plt.imshow(npframe/256)
        plt.plot(x_data, func(x_data, *popt), 'g--',
              label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt) + ', var=' + str(perr_mean))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        '''

        if(perr_mean > perr_mean_ant + delta_perr):
          data_xy = np.stack((x_data, y_data), axis=1)
          data_abs.append(data_xy)
          data_popt.append(popt)
          path_counter += 1

          print("data_abs: ")
          print(data_abs)
          print("path_counter: ")
          print(path_counter)
          print("data_popt: ")
          print(data_popt)

        perr_mean_ant = perr_mean
    ######################################

    blobber.draw_ball_path(frame)
    blobber.draw_ball(frame)
    cv.imwrite("frames/frame-{:03d}.jpg".format(n), frame)
    cv.imshow('frame', frame)
    if cv.waitKey(10) == 27:
      break
    n += 1
    frames.append(frame)
    npframe = np.array(frame)

  ############## Plotando todas as trajetórias
  colors = ['b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w']
  da_cnt = 0
  color_cnt = 0
  plt.imshow(npframe/256)
  for d_popt in data_popt:
    if (d_popt[0] > 0 and len(data_abs[da_cnt][:,0])>10):
      new_x_data = np.arange(min(data_abs[da_cnt][:,0])-100, max(data_abs[da_cnt][:,0])+100, 1)
      plt.plot(new_x_data, func(new_x_data, *d_popt), color=colors[color_cnt], label=str(da_cnt)+", total pts: "+str(len(data_abs[da_cnt][:,0])))
      color_cnt += 1
    da_cnt += 1
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  plt.show()
  ######################################

  #print("Saving GIF file")
  #with imageio.get_writer("smiling.gif", mode="I") as writer:
  #    for idx, frame in enumerate(frames):
  #        print("Adding frame to GIF file: ", idx + 1)
  #        writer.append_data(frame)


#test_clip("D:/Videos/aus4.avi")
test_clip(sys.argv[1])
