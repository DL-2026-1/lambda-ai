module Types where

type Weight = (SingleInput -> Double)

type Weights = [Weight]

type Bias = (SingleInput -> Double)

type ActivationFunction = ((Double -> Double), (Double -> Double)) 

type LearnRate = (SingleInput -> Double)

type Result = Double

type Results = [Result]

type Target = Double

type Targets = [Target]

type SingleInput = Double

type Input = [SingleInput]

type WeightSum = Double

type Gradient = Double

type Gradients = [Gradient]