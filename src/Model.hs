module Model where

import Layers

import Architeture
import Utils (chunksOf, scaleGradients)

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
    foldl (\m _ -> trainEpoch m dataset lossFunction minBatch) initialModel [1..epoch]

trainEpoch :: Model -> Dataset -> LossFunction -> MinBatch -> Model
trainEpoch model dataset lossFunction minBatch = 
    (foldl (trainBatch lossFunction) model . chunksOf minBatch) dataset 

trainBatch :: LossFunction -> Model -> Dataset -> Model
trainBatch lossFunction model batch = 
    foldl trainSample model batch
    where
        trainSample accModel sample =
            let (gradsAll, inps, resAll) = computePass accModel lossFunction 1.0 sample
            in compile accModel gradsAll inps resAll

computePass :: Model -> LossFunction -> Double -> (Inputs, Targets) -> (GradientsAll, Inputs, ResultsAll)
computePass model lossFunction scaleFactor (inputs, targets) =
    (backward model (scaleGradients scaleFactor (lossFunction targets (last resultsAll))) 
        resultsAll inputs, inputs, resultsAll)
    where
        resultsAll      = forward model inputs 