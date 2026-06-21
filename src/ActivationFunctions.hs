module ActivationFunctions where

import Types
import Data.List (sortOn)

type ThresholdValue = Double
type ThresholdValues = [ThresholdValue]
type Value = Double
type Values = [Value]

threshold :: ThresholdValues -> Values -> ActivationFunction
threshold thresholds inputs = (activationF, derivateF)
  where
    valuePairs = (sortOn fst . zip thresholds) inputs
    (maxValue, lastValue) = last valuePairs
    activationF v
        | v > maxValue = lastValue
        | otherwise = (snd . head . filter ((v>) . fst)) valuePairs
    derivateF _ = 0