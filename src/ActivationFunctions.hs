module ActivationFunctions where

import Types
import Data.List (sortOn)

type ThresholdValue = Double
type ThresholdValues = [ThresholdValue]
type Value = Double
type Values = [Value]

type Alpha = Double

threshold :: ThresholdValues -> Values -> ActivationFunction
threshold thresholds inputs = (activationF, derivateF)
  where
    valuePairs = (sortOn fst . zip thresholds) inputs
    (maxValue, lastValue) = last valuePairs
    activationF v
        | v > maxValue = lastValue
        | otherwise = (snd . head . filter ((v>) . fst)) valuePairs
    derivateF _ = 0

relu :: ActivationFunction
relu = (f, f') 
    where
        f x = max 0.0 x
        f' x = if x > 0.0 then 1.0 else 0.0

leakyRelu :: ActivationFunction
leakyRelu = (f, f')
    where
        f x = max (0.1 * x) x
        f' x = if x > 0 then 1 else 0.1

elu :: Alpha -> ActivationFunction
elu alpha = (f, f')
    where
        f  x = if x >= 0 then x else alpha * (exp x - 1)
        f' x = if x >= 0 then 1 else f x + alpha

tanhF :: ActivationFunction
tanhF = (f, f')
    where
        f    = tanh 
        f' x = 1 - (tanh x) ** 2

sigmoid :: ActivationFunction
sigmoid = (f, f')
    where
        f x  = 1 / (1 + exp (-x))
        f' x = f x * (1 - f x)