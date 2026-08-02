module MnistTest where

import Test.HUnit

import Layers (MetaLayer (MetaActivation), MetaLayer(MetaConvolution), Layer, genLayer')
import Convolutional (ConvolutionalLayer'(..))
import ActivationFunction (ActivationFunction'(..))
import Model (Model(Model))

mnistMetaLayers :: [MetaLayer]
mnistMetaLayers = 
    [
         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.005, filters     = 4, dimensions  = [9, 9]    })),
         (MetaActivation (ActivationFunction' "relu")),

         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.005, filters     = 8, dimensions  = [4, 11, 11]    })),
         (MetaActivation (ActivationFunction' "relu")),

         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.005, filters     = 10, dimensions  = [8, 10, 10]    })),
         (MetaActivation (ActivationFunction' "sigmoid"))

    ]

mnistModel :: IO (Model)
mnistModel = Model <$> mapM genLayer' mnistMetaLayers