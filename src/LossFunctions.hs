module LossFunctions where

import Architeture

simpleLF :: LossFunction
simpleLF (targets, dimTargets) (results, dimResults)
    | dimTargets /= dimResults = error "MSE need to have same dimensions"
    | otherwise                = (zipWith (-) targets results, dimTargets)