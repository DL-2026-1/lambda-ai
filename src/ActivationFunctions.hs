module ActivationFunctions where

import Types
import Data.List (sortOn)

type ThresholdValue = Double
type ThresholdValues = [ThresholdValue]
type Value = Double
type Values = [Value]

threshold :: ThresholdValues -> Values -> ActivationFunction
threshold thresholds inputs = 
    \v -> if v > maxValue 
          then lastValue 
          else (snd . head . filter ((v>) . fst)) valuePairs
    where 
        valuePairs = (sortOn fst . zip thresholds) inputs
        (maxValue, lastValue) = last valuePairs