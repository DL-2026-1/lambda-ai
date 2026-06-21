module MLP where

import Types

import Perceptron
import Data.List (zipWith4, transpose)

type Layer = [Perceptron]

type Layers = [Layer]

forwardLayer :: Input -> Layer -> Results
forwardLayer inputs layer =
    map (perceptronPredict inputs) layer

forward :: Layers -> Input -> [Results]
forward _ [] = []
forward [] _ = []
forward (layer:layers) inputs = results : forward layers results
    where
        results = forwardLayer inputs layer

trainMLP :: [(Input, Targets)] -> Layers -> Layers
trainMLP trainingData layers = 
    foldl (\l (inputs, targets) -> 
        learnMlP (inputs, targets) l) layers trainingData

learnMlP :: (Input, Targets) -> Layers -> Layers
learnMlP (inputs, targets) layers = fixMLP (inputs: results) layers (reverse gradients)
    where
        results = forward layers inputs
        gradients = backward inputs layers results targets 

fixMLP :: [Input] -> Layers -> [Gradients] -> Layers
fixMLP = zipWith3 fixLayer

fixLayer :: Input -> Layer -> Gradients -> Layer
fixLayer input = zipWith (fixPerceptron input) 

fixPerceptron :: Input -> Perceptron -> Gradient -> Perceptron
fixPerceptron input perceptron gradient = perceptron { weights = newWeights, bias = newBias }
    where
        learnRate' = learnRate perceptron
        newWeights = zipWith (\w -> fixWeightPerceptron w learnRate' gradient) (weights perceptron) input
        newBias = fixWeightPerceptron (bias perceptron) learnRate' gradient (biasInput perceptron)

fixWeightPerceptron :: Weight -> LearnRate -> Gradient -> SingleInput -> Weight
fixWeightPerceptron weight learnRate gradient input = 
    (* ((weight 1) + (learnRate (gradient * input))))

calculateGradientOutput :: Result -> Target -> ActivationFunction -> WeightSum -> Gradient
calculateGradientOutput result target f weightSum = 
    (target - result) * ((snd f) weightSum)

calculateGradient :: Gradients -> Weights -> ActivationFunction -> WeightSum -> Gradient
calculateGradient gradients weights f weightSum = 
    ((sum . zipWith (\w g -> w g) weights) gradients) * ((snd f) weightSum)

backward :: Input -> Layers -> [Results] -> Targets -> [Gradients]
backward _ _ _ [] = []
backward [] _ _ _ = []
backward _ _ [] _ = []
backward inputs layers results targets = gradientsOutput : backward' layers' (tail flow') gradientsOutput
    where
        flow = inputs : results
        layers' = reverse layers
        flow' = reverse flow
        gradientsOutput = calculateGradientsOutput (head flow') targets (head layers') ((head . tail) flow')

backward' :: Layers -> [Input] -> Gradients -> [Gradients]
backward' _ _ gs = [gs]
backward' [layerAfter, layerBefore] (_: input: _) gradients = 
    [calculateGradients gradients layerBefore layerAfter input]
backward' (layerAfter : layerBefore: layers) (_ : input : results) gradients =
    newGradients : backward' (layerBefore : layers) (input : results) newGradients
    where
        newGradients = calculateGradients gradients layerBefore layerAfter input 

calculateGradientsOutput :: Results -> Targets -> Layer -> Input -> Gradients
calculateGradientsOutput results targets layer input = 
    zipWith4 calculateGradientOutput results targets activationFunctions weightSums
     where 
        weightSums = map (sumWeight input) layer
        activationFunctions = map activationFunction layer

calculateGradients :: Gradients -> Layer -> Layer -> Input -> Gradients
calculateGradients gradients layerBefore layerAfter input = 
    zipWith4 calculateGradient (repeat gradients) weights' activationFunctions weightSums
    where
        weights' = (transpose . map weights) layerAfter
        activationFunctions = map activationFunction layerBefore
        weightSums = map (sumWeight input) layerBefore