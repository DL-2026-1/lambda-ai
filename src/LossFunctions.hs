module LossFunctions where

import Architeture

simpleLF :: LossFunction
simpleLF (targets, dimTargets) (results, dimResults)
    | dimTargets /= dimResults = error "MSE need to have same dimensions"
    | otherwise                = (zipWith (-) targets results, dimTargets)

mseLF :: LossFunction
mseLF (targets, dimTargets) (results, dimResults) 
    | length targets /= length results = error 
        $ "MSE need to have same dimensions. Dimensions targets: " 
            ++ show dimTargets ++ ". Dimensions results: " ++ show dimResults
            ++ " Length targets: " ++ show (length targets) ++ ". Length results: " ++ show (length results)
    | otherwise                = 
        ((map (/ n) . zipWith (\t r -> 2 * (r - t)) targets) results,
            dimTargets )
    where
        n = fromIntegral (length targets)