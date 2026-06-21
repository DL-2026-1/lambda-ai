module MyLib (
    ActivationFunction, 
    Perceptron(..), 
    Input,
    Target,
    perceptronPredict,
    trainPerceptron,
    sumWeight,
    fixWeightSinglePerceptron,
    learnPerceptron,
    threshold
    ) where

import Perceptron (
    Perceptron(..), 
    perceptronPredict, 
    sumWeight, 
    fixWeightSinglePerceptron, 
    learnPerceptron,
    trainPerceptron,
    )

import ActivationFunctions (
    threshold
    )

import Types (
    ActivationFunction,
    Input,
    Target
    )