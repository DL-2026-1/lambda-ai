module Model(
    Model(..), 
    forward, 
    backward, 
    compile, 
    train, 
    trainEpoch, 
    trainBatch, 
    computePass) where

import Layers
import Architeture
import Utils (chunksOf, scaleGradients, forcePass)

import Control.Parallel.Strategies (parMap, rseq) 

import Data.List (foldl')

data Model = Model [Layer]

forward :: Model -> Forward'
forward (Model layers) inputs =
    (tail 
    . scanl (\acc' fwd -> fwd acc') inputs 
    . map (fst . genLayer) 
    ) layers 

backward :: Model -> Backward'
backward (Model layers) initialGradients resultsAll inputs = 
    (tail 
    . scanr (\(bwd, inp) acc' -> bwd acc' inp) initialGradients  
    . zip (map (snd . genLayer) layers) 
    . (\results -> inputs : init results) 
    ) resultsAll

compile :: Model -> GradientsAll -> Inputs -> ResultsAll -> Model
compile (Model layers) gradientsAll inputs resultsAll = 
    Model $ updateLayers layers gradientsAll (inputs : init resultsAll)

evaluate :: Model -> Dataset -> LossFunction -> Double
evaluate model dataset lossFunction = 
    sum [ sum (fst (lossFunction targets (last (Model.forward model inputs)))) 
        | ((inputs, targets)) <- dataset ]

train :: Model -> Dataset -> Dataset -> LossFunction -> (Epoch, MinBatch) -> Model
train initialModel dataset validationSet lossFunction (epochs, minBatch) = 
    let initialValLoss = evaluate initialModel validationSet lossFunction        
        (_, _, bestModel) = foldl' (\(currentModel, bestLoss, bestModelAcc) _ ->
            let 
                trainedModel = trainEpoch currentModel dataset lossFunction minBatch
                valLoss = evaluate trainedModel validationSet lossFunction
            in 
                if valLoss < bestLoss
                    then (trainedModel, valLoss, trainedModel)
                    else (trainedModel, bestLoss, bestModelAcc)
            ) (initialModel, initialValLoss, initialModel) [1..epochs]
    in bestModel
    -- foldl' (\m _ -> trainEpoch m dataset lossFunction minBatch) initialModel [1..epoch]

trainEpoch :: Model -> Dataset -> LossFunction -> MinBatch -> Model
trainEpoch model dataset lossFunction minBatch = 
    (foldl' (trainBatch lossFunction) model . chunksOf minBatch) dataset 

trainBatch :: LossFunction -> Model -> Dataset -> Model
trainBatch lossFunction model batch = 
    let scaleFactor = 1.0 / fromIntegral (length batch)
        computedPasses = parMap rseq (\sample -> 
            forcePass (computePass model lossFunction scaleFactor sample)
            ) batch
    in foldl' (\accModel (gradsAll, inps, resAll) -> compile accModel gradsAll inps resAll) model computedPasses

computePass :: Model -> LossFunction -> Double -> (Inputs, Targets) -> (GradientsAll, Inputs, ResultsAll)
computePass model lossFunction scaleFactor (inputs, targets) =
    (backward model (scaleGradients scaleFactor (lossFunction targets (last resultsAll))) 
        resultsAll inputs, inputs, resultsAll)
    where
        resultsAll = forward model inputs