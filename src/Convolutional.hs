module Convolutional where

import System.Random

import Architeture

-- -----------------------------------------------------

data ConvolutionalPerceptron = ConvolutionalPerceptron 
  {
    kernel             :: Weights
  , bias               :: Weight
  , learnRate          :: Weight
  }

data ConvolutionalLayer = ConvolutionalLayer 
  {
    convolutionalPerceptrons :: [ConvolutionalPerceptron]
  }

-- -----------------------------------------------------

genKernell :: Dimensions -> IO (Weights)
genKernell dimensions = do
  let size = product dimensions
  weights <- mapM (\_ -> randomRIO (-0.5, 0.5)) [1..size]
  return (weights, dimensions)

genCNNLayer :: (Weight, Int, Dimensions) -> IO (ConvolutionalLayer)
genCNNLayer (learnRate, filters, dimensions) = do
  kernels <- mapM (\_ -> genKernell dimensions) [1..filters]
  learnRates <- mapM (\_ -> randomRIO (0.01, 0.1)) [1..filters]
  biases <- mapM (\_ -> randomRIO (-0.5, 0.5)) [1..filters]
  let learnRates' = if learnRate > 0 then replicate filters learnRate else learnRates 
  return ConvolutionalLayer 
    { 
      convolutionalPerceptrons = zipWith3 ConvolutionalPerceptron kernels biases learnRates'
    }

-- -----------------------------------------------------

genCNNPerceptronLast :: ConvolutionalPerceptron -> (Forward, Backward)
genCNNPerceptronLast convolutionalPerceptron = (forward, backward)
  where
    forward :: Forward
    forward inputs = (map (+(bias convolutionalPerceptron)) weightSum, dimensions)
      where 
        (weightSum, dimensions) =  convolutionWrapper inputs (kernel convolutionalPerceptron) []
    
    backward :: Backward
    backward (gradients, gradientsDimensions) _ = 
      convolutionWrapper (gradients, gradientsDimensions) (flipN (kernel convolutionalPerceptron)) fullPaddings
      where
        fullPaddings = map pred (snd (kernel convolutionalPerceptron))
        
updateCNNPerceptron :: ConvolutionalPerceptron -> Gradients -> Inputs -> ConvolutionalPerceptron
updateCNNPerceptron convolutionalPerceptron gradients inputs = 
  convolutionalPerceptron { kernel = (
                              (zipWith (+) ((fst . kernel) convolutionalPerceptron) . map (* (learnRate convolutionalPerceptron))) newKernel, 
                              newDimensions), 
                            bias = bias convolutionalPerceptron + ((* (learnRate convolutionalPerceptron)) . sum . fst) gradients
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
        combinedResults = concatMap fst resList
        numFilters      = length perceptrons
        outDims         = snd (head resList)

    backwardLayer :: Backward
    backwardLayer gradients inputs = (totalDX, snd inputs)
      where
        gradSlices = splitGradients gradients
        dXList     = zipWith (\bwd g -> fst (bwd g inputs)) backwards gradSlices
        totalDX    = foldl1 (zipWith (+)) dXList

updateCNNLayer :: ConvolutionalLayer -> Gradients -> Inputs -> ConvolutionalLayer
updateCNNLayer layer gradients inputs = layer { convolutionalPerceptrons = updatedPs }
  where
    updatedPs   = 
      zipWith (\p g -> updateCNNPerceptron p g inputs) (convolutionalPerceptrons layer) (splitGradients gradients)