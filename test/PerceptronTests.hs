module PerceptronTests where

import Test.HUnit

import MyLib 

myThreshold :: ActivationFunction
myThreshold = threshold [-100, 0.0] [-1.0, 1.0]

myPerceptron :: Perceptron
myPerceptron = Perceptron {
    weights = [(* 0.4), (* (-0.6)), (* 0.6)],
    bias = (* 0.5),
    biasInput = (-1.0),
    activationFunction = myThreshold,
    learnRate = (* 0.4)
}

perceptronAfterLearning :: Perceptron
perceptronAfterLearning = Perceptron {
    weights = [(* 1.2), (* 0.2), (* (-0.2))],
    bias = (* 0.5),
    biasInput = (-1.0),
    activationFunction = myThreshold,
    learnRate = (* 0.4)
}

perceptronData :: [(Input, Target)]
perceptronData = [
    ([0.0, 0.0, 1.0], -1.0),
    ([1.0, 1.0, 0.0], 1.0)
    ]

perceptronPredictData :: [(Input, Target)]
perceptronPredictData = [
    ([1.0, 1.0, 1.0], 1.0),
    ([0.0, 0.0, 0.0], -1.0),
    ([1.0, 0.0, 0.0], 1.0),
    ([0.0, 1.0, 1.0], -1.0)
    ]

testLearnPerceptron :: Test
testLearnPerceptron = TestCase (
    assertEqual 
    "learnPerceptron should update the perceptron correctly" perceptronAfterLearning 
    (trainPerceptron perceptronData myPerceptron)
    )

testPredictPerceptron :: Test
testPredictPerceptron = TestCase (
    assertEqual 
    "perceptronPredict should return correct predictions" perceptronPredictData
    (map (\(inputs, _) -> (inputs, perceptronPredict inputs (trainPerceptron perceptronData myPerceptron))) perceptronPredictData)
    )   

testAllPerceptron :: Test
testAllPerceptron = TestList [testLearnPerceptron, testPredictPerceptron]