module MyLib (
    ActivationFunction, 
    Perceptron(..), 
    Input,
    Target,
    perceptronPredict,
    trainPerceptron,
    sumWeight,
    fixWeight,
    learnPerceptron,
    threshold
    ) where

import Perceptron (
    Perceptron(..), 
    perceptronPredict, 
    sumWeight, 
    fixWeight, 
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