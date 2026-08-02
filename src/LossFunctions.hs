module LossFunctions where

import Architeture

simpleLF :: LossFunction
simpleLF (targets, dimTargets) (results, dimResults)
    | dimTargets /= dimResults = error "MSE need to have same dimensions"
    | otherwise                = (zipWith (-) targets results, dimTargets)

mseLF :: LossFunction
mseLF (targets, dimTargets) (results, dimResults) 
    | dimTargets /= dimResults = error "MSE need to have same dimensions"
    | otherwise                = 
        ((map (/ ((fromIntegral . length) targets)) -- divide for targets dimensions
        . zipWith (\t r -> (r - t) ^ (2 :: Integer)) targets) results, -- square diff 
            dimTargets )