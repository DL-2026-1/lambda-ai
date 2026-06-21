module Perceptron where

import Types 

import Utils (aproxEq)

data Perceptron = Perceptron
  { weights :: [Weight]
  , bias :: Bias
  , biasInput :: SingleInput
  , activationFunction :: ActivationFunction
  , learnRate :: LearnRate
  }

perceptronPredict :: [Double] -> Perceptron -> Double
perceptronPredict inputs neuron = (fst (activationFunction neuron)) $ sumWeight inputs neuron + ((bias neuron) (biasInput neuron))

sumWeight :: [Double] -> Perceptron -> Double
sumWeight inputs neuron = (sum . zipWith (\w i -> w i) (weights neuron)) inputs

fixWeight :: Weight -> LearnRate -> SingleInput -> Target -> Result -> Weight
fixWeight weight learnRate input target result = 
    (* (weight 1 + learnRate (input * (target - result))))

learnPerceptron :: Input -> Target -> Perceptron -> Perceptron
learnPerceptron inputs target neuron = neuron { weights = newWeights, bias = newBias }
  where
    prediction = perceptronPredict inputs neuron
    newWeights = zipWith (\w i -> fixWeight w (learnRate neuron) i target prediction) (weights neuron) inputs
    newBias = fixWeight (bias neuron) (learnRate neuron) (-1) target prediction

trainPerceptron :: [(Input, Target)] -> Perceptron -> Perceptron
trainPerceptron trainingData neuron = 
  foldl (\n (inputs, target) -> 
    learnPerceptron inputs target n) neuron trainingData

instance Eq Perceptron where
    (==) p1 p2 = 
        and [ all (\(w1, w2) -> aproxEq (w1 1) (w2 1)) (zip (weights p1) (weights p2))
            , aproxEq (bias p1 1) (bias p2 1)
            , biasInput p1 == biasInput p2
            , aproxEq (learnRate p1 1) (learnRate p2 1)
            ]

instance Show Perceptron where
    show neuron = "Perceptron { weights = " ++ show (map (\w -> w 1) (weights neuron)) ++ 
                    ", bias = " ++ show (bias neuron 1) ++ 
                    ", biasInput = " ++ show (biasInput neuron) ++ 
                    ", learnRate = " ++ show (learnRate neuron 1) ++ " }"