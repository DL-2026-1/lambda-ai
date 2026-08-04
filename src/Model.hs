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

train :: Model -> Dataset -> LossFunction -> (Epoch, MinBatch) -> Model
train initialModel dataset lossFunction (epochs, minBatch) = 
    foldl' (\m _ -> trainEpoch m dataset lossFunction minBatch) initialModel [1..epochs]

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