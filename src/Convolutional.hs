module ConvolutionalPerceptron where

import Architeture

data ConvolutionalPerceptron = ConvolutionalPerceptron 
  {
    kernel             :: Weights
  , bias               :: Weight
  , learnRate          :: Weight
  }

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