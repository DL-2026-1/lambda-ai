module Model where

import Layers

import Architeture

data Model = Model [Layer]

forward :: Model -> Forward'
forward (Model layers) inputs = 
    (tail -- Ignore the initial input, which is not a result of any layer
    . scanl (\acc' fwd -> fwd acc') inputs -- Accumulate the results of each layer's forward pass
    . map (fst . genLayer) -- Get the forward function for each layer
    ) layers 

backward :: Model -> Backward'
backward (Model layers) initialGradients resultsAll inputs = 
    (tail -- Ignore the initial gradients, which correspond to the input and are not updated
    . scanr (\(bwd, inp) acc' -> bwd acc' inp) initialGradients  -- Accumulate the gradients from each layer's backward pass
    . zip (map (snd . genLayer) layers) -- Get the backward function for each layer
    . (\results -> inputs : init results) -- Pair each backward function with the corresponding input to that layer's forward pass
    ) resultsAll

compile :: Model -> GradientsAll -> Inputs -> ResultsAll -> Model
compile (Model layers) gradientsAll inputs resultsAll = 
    Model $ updateLayers layers gradientsAll (inputs : init resultsAll) -- Update each layer with its corresponding gradients and inputs

train :: Model -> Dataset  -> LossFunction -> (Epoch, MinBatch) -> Model
train initialModel dataset lossFunction (epoch, minBatch) = 
    foldl (\m _ -> trainEpoch m dataset lossFunction) initialModel [1..epoch]

trainEpoch :: Model -> Dataset -> LossFunction-> Model
trainEpoch model dataset lossFunction = foldl (trainStep lossFunction) model dataset 

trainStep :: LossFunction -> Model ->  (Inputs, Targets) -> Model
trainStep lossFunction model (inputs, targets) = compile model gradsAll inputs resultsAll
    where
        resultsAll = forward model inputs 
        finalResult = last resultsAll
        initialGradient = lossFunction targets finalResult
        gradsAll = backward model initialGradient resultsAll inputs