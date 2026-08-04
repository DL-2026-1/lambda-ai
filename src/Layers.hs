{-# LANGUAGE DeriveGeneric #-}
module Layers(
    Layer(..), 
    MetaLayer(..), 
    updateLayer, 
    updateLayers, 
    genLayer', 
    genLayer) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData, force)


import Convolutional ( ConvolutionalLayer, updateCNNLayer, ConvolutionalLayer', genCNNLayer, genCNNLayerFunctions )

import Architeture 
import ActivationFunction

data Layer = ConvolutionalLayer Convolutional.ConvolutionalLayer | ActivationLayer Architeture.ActivationFunction
    deriving (Generic)
instance NFData Layer

data MetaLayer = MetaConvolution Convolutional.ConvolutionalLayer' | MetaActivation ActivationFunction'

-- -----------------------------------------------------------------------

updateLayer :: Layer -> Gradients -> Inputs -> Layer
updateLayer (ConvolutionalLayer convLayer) gradients inputs = ConvolutionalLayer (updateCNNLayer convLayer gradients inputs)
updateLayer (ActivationLayer actFunc) _ _ = ActivationLayer actFunc

updateLayers :: [Layer] -> GradientsAll -> InputsAll -> [Layer]
updateLayers layers gradientsAll inputsAll = zipWith3 updateLayer layers gradientsAll inputsAll

-- -----------------------------------------------------------------------

genLayer' :: MetaLayer -> IO Layer
genLayer' (MetaConvolution convLayer') = ConvolutionalLayer <$> genCNNLayer convLayer'
genLayer' (MetaActivation actFunc') = ActivationLayer <$> return (genActivationLayer' actFunc')

------------------------------------------------------------------------

genLayer :: Layer -> (Forward, Backward)
genLayer (Layers.ConvolutionalLayer convLayer)       = genCNNLayerFunctions convLayer
genLayer (Layers.ActivationLayer activationFunction) = genActivationLayer activationFunction