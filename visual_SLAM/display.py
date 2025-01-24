# PyGame > SDL2 > OpenCV
import pygame
from pygame.locals import DOUBLEBUF
import numpy as np

class Display(object):
  def __init__(self, W, H):
    pygame.init()
    self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
    self.surface = pygame.Surface(self.screen.get_size()).convert()

  def display2D(self, img):
    pygame.surfarray.blit_array(self.surface, np.stack((img, img, img), axis=-1).swapaxes(0,1))
    self.screen.blit(self.surface, (0,0))
    pygame.display.flip()