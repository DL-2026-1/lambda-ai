module IrisTest where

import Test.HUnit
import Data.List (maximumBy)
import Data.Ord (comparing)

import Layers (MetaLayer (MetaActivation, MetaConvolution), genLayer')
import Convolutional (ConvolutionalLayer'(..))
import ActivationFunction (ActivationFunction'(..))
import Model (Model(Model), train, forward)
import Utils (readCSV, stratifiedSplit)
import LossFunctions (mseLF)
import Architeture (Inputs, Targets, Dataset)
import GHC.IO (unsafePerformIO)

irisMetaLayers :: [MetaLayer]
irisMetaLayers = 
    [
         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.01, filters     = 8, dimensions  = [4]    })),
         (MetaActivation (ActivationFunction' "relu")),

         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.01, filters     = 16, dimensions  = [8,1]    })),
         (MetaActivation (ActivationFunction' "relu")),

         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.01, filters     = 3, dimensions  = [16, 1, 1]    })),
         (MetaActivation (ActivationFunction' "sigmoid"))
    ]

irisModel :: IO Model
irisModel = Model <$> mapM genLayer' irisMetaLayers

toOneHot :: Double -> [Double]
toOneHot 0.0 = [1.0, 0.0, 0.0]
toOneHot 1.0 = [0.0, 1.0, 0.0]
toOneHot 2.0 = [0.0, 0.0, 1.0]
toOneHot _   = [0.0, 0.0, 0.0]

argmax :: Ord a => [a] -> Int
argmax xs = fst $ maximumBy (comparing snd) (zip [0..] xs)

csvData :: IO [[Double]]
csvData = readCSV "data/iris.csv"

dataset :: IO Dataset
dataset = do
    csv <- csvData
    return $ 
        map (\row -> 
        let features = take 4 row
            label = last row
            inputs = (features, [4])
            targets = (toOneHot label, [3,1,1,1])
        in (inputs, targets)) csv

trainSet :: Dataset
testSet :: Dataset
(trainSet, testSet) = unsafePerformIO $ do
    ds <- dataset
    stratifiedSplit 0.8 ds

trainedModel :: Model
trainedModel =  
    unsafePerformIO $ do
    initialModel <- irisModel
    let epochs = 30
        minBatch = 20
    return $ train initialModel trainSet mseLF (epochs, minBatch)

evaluate :: (Inputs, Targets) -> Double
evaluate (inputs, targets) =
            let resultsAll = forward trainedModel inputs
                finalResult = fst (last resultsAll)
                predictedClass = argmax finalResult
                actualClass = argmax (fst targets)
            in if predictedClass == actualClass then 1.0 else 0.0
        
correctPredictions :: Double
correctPredictions = sum (map evaluate testSet)

totalPredictions :: Double
totalPredictions = fromIntegral (length testSet)

accuracy :: Double
accuracy = correctPredictions / totalPredictions

testIrisMinimum :: Test
testIrisMinimum = TestCase (assertBool ("O modelo nao obteve precisao superior a 1/3. Acuracia obtida: " ++ show accuracy) (accuracy > (1.0 / 3.0)))

testIrisMedium :: Test
testIrisMedium = TestCase (assertBool ("O modelo nao obteve precisao superior a 1/2. Acuracia obtida: " ++ show accuracy) (accuracy > 0.5))

testIrisHigh :: Test
testIrisHigh = TestCase (assertBool ("O modelo nao obteve precisao superior a 2/3. Acuracia obtida: " ++ show accuracy) (accuracy > (2.0 / 3.0)))

testIris :: Test
testIris = TestList [TestLabel "Verificacao de Acuracia Minima - Dataset Iris" testIrisMinimum,
                     TestLabel "Verificacao de Acuracia Media - Dataset Iris" testIrisMedium,
                     TestLabel "Verificacao de Acuracia Alta - Dataset Iris" testIrisHigh]