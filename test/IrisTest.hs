module IrisTest where

import Test.HUnit

import Layers (MetaLayer (MetaActivation), MetaLayer(MetaConvolution), Layer, genLayer')
import Convolutional (ConvolutionalLayer'(..))
import ActivationFunction (ActivationFunction'(..))
import Model (Model(Model))

irisMetaLayers :: [MetaLayer]
irisMetaLayers = 
    [
         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.01, filters     = 8, dimensions  = [2]    })),
         (MetaActivation (ActivationFunction' "relu")),

         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.01, filters     = 16, dimensions  = [8,2]    })),
         (MetaActivation (ActivationFunction' "relu")),

         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.01, filters     = 3, dimensions  = [16, 2]    })),
         (MetaActivation (ActivationFunction' "sigmoid"))

    ]

irisModel :: IO (Model)
irisModel = Model <$> mapM genLayer' irisMetaLayers