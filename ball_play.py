import numpy as np
import cv2 as cv
import os
import ball_net as bn
import blobber
import sys
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit

import time

################ Parabolic function to approximate the trajectory
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

  ###### Variables for calculating the trajectory
  x_data = [] # X data for the trajectory
  y_data = [] # Y data for the trajectory
  data_abs = [[]] # List of points [[[x1,y1],...,[xn,yn]],...] for each detected trajectory
  data_popt = [[]] # List of parameters [[a,b,c],...] of the line fitting for the trajectory
  path_counter = 0 # Keep track of how many trajectories were detected
  delta_perr = 0.1 # Represent how much the difference in line fitting error is necessary for a new path identification
  perr_mean = 0 # Keep the value of the covariance mean
  perr_mean_ant = -delta_perr # Keep the last value of the line fitting error
  frames_saved = [] # List of frames for keep track of the initial and end frames of trajectories
  ###############################################################

  while(True):
    ret, frame = vs.read()
    if not ret or frame is None:
      break

    h = frame.shape[0]
    w = frame.shape[1]

    frame = cv.resize(frame, (int(w/2),int(h/2)))
    mask = backSub.apply(frame)

    mask = cv.dilate(mask, None)
    mask = cv.GaussianBlur(mask, (15, 15),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)

    prev_bb = blobber.handle_blobs(mask, frame)

    ############## Trajectory detection
    x_data = []
    y_data = []

    if prev_bb != None and len(prev_bb.pts)>2:
      for point in prev_bb.pts:
        if not point in data_abs[path_counter] and np.mean(np.array(prev_bb.pts)[:,1]) < 200:
          x_data.append(point[0])
          y_data.append(point[1])

      if len(x_data)>2:
        try:
          popt, pcov = curve_fit(func, x_data, y_data, maxfev=5000)
          perr = np.sqrt(np.diag(pcov))
          perr_mean = np.mean(perr)
        except:
          continue

        # End of trajectory detected
        if(perr_mean > perr_mean_ant + delta_perr and perr_mean < 10 and np.mean(np.array(y_data) < 180)): # Net height ~180
          data_xy = np.stack((x_data, y_data), axis=1)
          data_abs.append(data_xy)
          data_popt.append(popt)
          path_counter += 1

          frames_saved.append(frame)

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

  ############## Combining trajectories if parameters are similar
  print("data_abs before: ")
  print(data_abs)
  da_cnt = 1
  delta = 0.2
  for d_popt_num in range(1,len(data_popt)-1):
    a_curr = data_popt[d_popt_num][0]
    a_next = data_popt[d_popt_num+1][0]
    b_curr = data_popt[d_popt_num][1]
    b_next = data_popt[d_popt_num+1][1]
    c_curr = data_popt[d_popt_num][2]
    c_next = data_popt[d_popt_num+1][2]
    if ((b_next < b_curr+(b_curr*delta) and b_next > b_curr-(b_curr*delta))
    and (c_next < c_curr+(c_curr*delta) and c_next > c_curr-(c_curr*delta))):
      print("Here! At d_popt_num = " + str(d_popt_num))
      if (len(data_abs[d_popt_num]) > len(data_abs[d_popt_num+1])):
        data_abs = np.delete(data_abs, d_popt_num+1)
        data_popt = np.delete(data_popt, d_popt_num+1)
      else:
        data_abs = np.delete(data_abs, d_popt_num)
        data_popt = np.delete(data_popt, d_popt_num)
      da_cnt += 1
    da_cnt += 1
    if (da_cnt >= len(data_popt)): break
  print("data_abs after: ")
  print(data_abs)
  ######################################

  ############## Plotting trajectories
  colors = ['b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w']
  da_cnt = 0
  color_cnt = 0
  plt.imshow(npframe/256)
  for d_popt in data_popt:
    if (da_cnt != 0 and d_popt[0] > 0 and len(data_abs[da_cnt][:,0])>3):
      x_data = data_abs[da_cnt][:,0]
      if x_data[-1]-x_data[-2] > 0: new_x_data = np.arange(x_data[-1], x_data[-1]+10*(x_data[-1]-x_data[-2]), 1)
      if x_data[-1]-x_data[-2] < 0: new_x_data = np.arange(x_data[-1], x_data[-1]+10*(x_data[-1]-x_data[-2]), -1)
      plt.plot(x_data, func(x_data, *d_popt), color=colors[color_cnt], label=str(color_cnt+1)+" - total points: "+str(len(data_abs[da_cnt][:,0])))
      plt.plot(new_x_data, func(new_x_data, *d_popt), colors[color_cnt]+'--')
      color_cnt += 1
    da_cnt += 1
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  plt.show()

  for frame_saved_num in range(len(frames_saved)-1):
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    npframe1 = np.array(frames_saved[frame_saved_num])
    npframe2 = np.array(frames_saved[frame_saved_num+1])
    axs[0].imshow(npframe1/256)
    axs[1].imshow(npframe2/256)
    plt.show()
  ######################################

  #print("Saving GIF file")
  #with imageio.get_writer("smiling.gif", mode="I") as writer:
  #    for idx, frame in enumerate(frames):
  #        print("Adding frame to GIF file: ", idx + 1)
  #        writer.append_data(frame)


#test_clip("D:/Videos/aus4.avi")
test_clip(sys.argv[1])
