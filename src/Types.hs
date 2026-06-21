module Types where

type Weight = (SingleInput -> Double)

type Bias = (SingleInput -> Double)

type ActivationFunction = ((Double -> Double), (Double -> Double)) 

type LearnRate = (SingleInput -> Double)

type Result = Double

type Target = Double

type SingleInput = Double

type Input = [SingleInput]