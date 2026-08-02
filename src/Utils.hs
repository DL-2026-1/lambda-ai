module Utils where

import System.Random
import Data.List (sortOn, groupBy)
import Data.Function (on)

import Architeture 

readCSV :: FilePath -> IO [[Double]]
readCSV path = do
    content <- readFile path
    let rows = lines content        
        splitComma :: String -> [String]
        splitComma [] = [""]
        splitComma (c:cs)
            | c == ','  = "" : rest
            | otherwise = (c : head rest) : tail rest
            where rest = splitComma cs
        parseRow = map read . splitComma
    return $ map parseRow (filter (not . null) rows)

shuffle :: [a] -> IO [a]
shuffle [] = return []
shuffle xs = do
    randomTags <- mapM (\_ -> randomIO :: IO Int) xs
    return $ map snd $ sortOn fst (zip randomTags xs)

stratifiedSplit :: Double -> [(Inputs, Results)] -> IO ([(Inputs, Results)], [(Inputs, Results)])
stratifiedSplit trainRatio dataset = do
    let sortedData = sortOn snd dataset
        groups     = groupBy ((==) `on` snd) sortedData
    splitGroups <- mapM processGroup groups
    let (trainSets, testSets) = unzip splitGroups
    finalTrain <- shuffle (concat trainSets)
    finalTest  <- shuffle (concat testSets)
    return (finalTrain, finalTest)
  where
    processGroup :: [(Inputs, Results)] -> IO ([(Inputs, Results)], [(Inputs, Results)])
    processGroup group = do
        shuffled <- shuffle group
        let trainSize = round (trainRatio * fromIntegral (length shuffled))
        return $ splitAt trainSize shuffled

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

scaleGradients :: Double -> Gradients -> Gradients
scaleGradients factor (grads, dims) = (map (* factor) grads, dims)