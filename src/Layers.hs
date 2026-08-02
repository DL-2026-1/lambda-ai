module Layers where

import Convolutional

import Architeture 

data Layer = ConvolutionalLayer' ConvolutionalLayer
           | ActivationLayer' ActivationFunction

updateLayer :: Layer -> Gradients -> Inputs -> Layer
updateLayer (ConvolutionalLayer' convLayer) gradients inputs = ConvolutionalLayer' (updateCNNLayer convLayer gradients inputs)
updateLayer (ActivationLayer' actFunc) _ _ = ActivationLayer' actFunc

updateLayers :: [Layer] -> GradientsAll -> InputsAll -> [Layer]
updateLayers layers gradientsAll inputsAll = zipWith3 updateLayer layers gradientsAll inputsAll