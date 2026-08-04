{-# LANGUAGE DeriveGeneric #-}

module Convolutional(
  ConvolutionalPerceptron(..), 
  ConvolutionalLayer(..), 
  ConvolutionalLayer'(..), 
  genKernell, 
  genCNNLayer, 
  genCNNPerceptronLast, 
  updateCNNPerceptron, 
  splitGradients, 
  genCNNLayerFunctions, 
  updateCNNLayer) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData, force)

import System.Random
import Data.List (foldl1')

import Architeture

-- -----------------------------------------------------

data ConvolutionalPerceptron = ConvolutionalPerceptron 
  {
    kernel             :: Weights
  , bias               :: Weight
  , learnRate          :: Weight
  }
  deriving (Show, Generic)
instance NFData ConvolutionalPerceptron

data ConvolutionalLayer = ConvolutionalLayer 
  {
    convolutionalPerceptrons :: [ConvolutionalPerceptron]
  }
  deriving (Show, Generic)
instance NFData ConvolutionalLayer

data ConvolutionalLayer' = ConvolutionalLayer' 
  {
    learnRate'  :: Weight
  , filters     :: Int
  , dimensions  :: Dimensions    
  }

-- -----------------------------------------------------

genKernell :: Dimensions -> IO (Weights)
genKernell dims = do
  let size = product dims
  weights <- mapM (\_ -> randomRIO (-0.01, 0.01)) [1..size]
  return (weights, dims)

genCNNLayer :: ConvolutionalLayer' -> IO (ConvolutionalLayer)
genCNNLayer convolutional' = do
  kernels <- mapM (\_ -> genKernell (dimensions convolutional')) [1..(filters convolutional')]
  learnRates <- mapM (\_ -> randomRIO (0.01, 0.1)) [1..(filters convolutional')]
  biases <- mapM (\_ -> randomRIO (-0.01, 0.01)) [1..(filters convolutional')]
  let learnRates' = if (learnRate' convolutional') > 0 then replicate (filters convolutional') (learnRate' convolutional') else learnRates 
  return ConvolutionalLayer 
    { 
      convolutionalPerceptrons = zipWith3 ConvolutionalPerceptron kernels biases learnRates'
    }

-- -----------------------------------------------------

genCNNPerceptronLast :: ConvolutionalPerceptron -> (Forward, Backward)
genCNNPerceptronLast convolutionalPerceptron = (forward, backward)
  where
    forward :: Forward
    forward inputs = (map (+(bias convolutionalPerceptron)) weightSum, outputDimensions)
      where 
        (weightSum, outputDimensions) =  convolutionWrapper inputs (kernel convolutionalPerceptron) []
    
    backward :: Backward
    backward (gradients, gradientsDimensions) _ = 
      convolutionWrapper (gradients, gradientsDimensions) (flipN (kernel convolutionalPerceptron)) fullPaddings
      where
        fullPaddings = map pred (snd (kernel convolutionalPerceptron))
        
updateCNNPerceptron :: ConvolutionalPerceptron -> Gradients -> Inputs -> ConvolutionalPerceptron
updateCNNPerceptron convolutionalPerceptron gradients inputs = 
  convolutionalPerceptron { kernel = (
                              (zipWith (-) ((fst . kernel) convolutionalPerceptron) . map (* (learnRate convolutionalPerceptron))) newKernel, 
                              newDimensions), 
                            bias = bias convolutionalPerceptron - ((* (learnRate convolutionalPerceptron)) . sum . fst) gradients
                            }  
    where
      (newKernel, newDimensions) = convolutionWrapper inputs gradients []

-- -----------------------------------------------------

splitGradients :: Gradients -> [Gradients]
splitGradients (gradientsArray, (nFilters : dimensionsPerArray)) =
  [(take singleMapSize (drop (i * singleMapSize) gradientsArray), dimensionsPerArray) 
  | i <- [0 .. nFilters - 1]]
  where 
    singleMapSize = product dimensionsPerArray
splitGradients (gradientsArray, dims) = [(gradientsArray, dims)]

genCNNLayerFunctions :: ConvolutionalLayer -> (Forward, Backward)
genCNNLayerFunctions layer = (forwardLayer, backwardLayer)
  where
    perceptrons = convolutionalPerceptrons layer
    nodes       = map genCNNPerceptronLast perceptrons
    forwards    = map fst nodes
    backwards   = map snd nodes

    forwardLayer :: Forward
    forwardLayer inputs = (combinedResults, numFilters : outDims)
      where
        resList         = map ($ inputs) forwards
        firstRes = case resList of
            (result:_) -> result
            []         -> error "Convolutional layer has no perceptrons"
        combinedResults = concatMap fst resList
        numFilters      = length perceptrons
        outDims         = snd firstRes

    backwardLayer :: Backward
    backwardLayer gradients inputs = (totalDX, snd inputs)
      where
        gradSlices = splitGradients gradients
        dXList     = zipWith (\bwd g -> fst (bwd g inputs)) backwards gradSlices
        totalDX    = foldl1' (zipWith (+)) dXList

updateCNNLayer :: ConvolutionalLayer -> Gradients -> Inputs -> ConvolutionalLayer
updateCNNLayer layer gradients inputs = layer { convolutionalPerceptrons = updatedPs }
  where
    updatedPs   = 
      zipWith (\p g -> updateCNNPerceptron p g inputs) (convolutionalPerceptrons layer) (splitGradients gradients)