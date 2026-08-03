module MnistTest where

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

mnistMetaLayers :: [MetaLayer]
mnistMetaLayers = 
    [
         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.01, filters = 8, dimensions = [19, 19] })),
         (MetaActivation (ActivationFunction' "relu")),

         (MetaConvolution (ConvolutionalLayer' { learnRate' = 0.01, filters = 10, dimensions = [8, 10, 10] })),
         (MetaActivation (ActivationFunction' "sigmoid"))
    ]

mnistModel :: IO Model
mnistModel = Model <$> mapM genLayer' mnistMetaLayers

toOneHot :: Double -> [Double]
toOneHot label = [if label == l then 1.0 else 0.0 | l <- [0.0 .. 9.0]]

argmax :: Ord a => [a] -> Int
argmax xs = fst $ maximumBy (comparing snd) (zip [0..] xs)

csvData :: IO [[Double]]
csvData = readCSV "data/mnist_train.csv"

csvTestData :: IO [[Double]]
csvTestData = readCSV "data/mnist_test.csv"

dataset :: IO Dataset
dataset = do
    csv <- csvData
    return $ 
        map (\row -> 
        let label    = head row
            features = map (/ 255.0) (tail row) 
            inputs   = (features, [28, 28])
            targets  = (toOneHot label, [10, 1, 1, 1])
        in (inputs, targets)) csv

datasetTest :: IO Dataset
datasetTest = do
    csv <- csvTestData
    return $ 
        map (\row -> 
        let label    = head row
            features = map (/ 255.0) (tail row) 
            inputs   = (features, [28, 28])
            targets  = (toOneHot label, [10, 1, 1, 1])
        in (inputs, targets)) csv


trainSet :: Dataset
testSet :: Dataset
(trainSet, testSet) = unsafePerformIO $ do
    ds <- dataset
    stratifiedSplit 0.8 ds

testSet' = unsafePerformIO $ datasetTest

trainedModel :: Model
trainedModel =  
    unsafePerformIO $ do
    initialModel <- mnistModel
    let epochs = 5
        minBatch = 32
    return $ train initialModel trainSet mseLF (epochs, minBatch)

evaluate :: (Inputs, Targets) -> Double
evaluate (inputs, targets) =
            let resultsAll     = forward trainedModel inputs
                finalResult    = fst (last resultsAll)
                predictedClass = argmax finalResult
                actualClass    = argmax (fst targets)
            in if predictedClass == actualClass then 1.0 else 0.0
    
correctPredictions :: Double
correctPredictions = sum (map evaluate testSet)

correctPredictionsTest :: Double
correctPredictionsTest = sum (map evaluate testSet')

totalPredictions :: Double
totalPredictions = fromIntegral (length testSet)

totalPredictionsTest :: Double
totalPredictionsTest = fromIntegral (length testSet')

accuracy :: Double
accuracy = correctPredictions / totalPredictions

accuracyTest :: Double
accuracyTest = correctPredictionsTest / totalPredictionsTest

testMnistMinimum :: Test
testMnistMinimum = TestCase (assertBool ("O modelo nao obteve precisao superior a 10%. Acuracia obtida: " ++ show accuracy) (accuracy > 0.1))

testMnistLow :: Test
testMnistLow = TestCase (assertBool ("O modelo nao obteve precisao superior a 20%. Acuracia obtida: " ++ show accuracy) (accuracy > 0.2))

testMnistMedium :: Test
testMnistMedium = TestCase (assertBool ("O modelo nao obteve precisao superior a 40%. Acuracia obtida: " ++ show accuracy) (accuracy > 0.4))

testMnistHigh :: Test
testMnistHigh = TestCase (assertBool ("O modelo nao obteve precisao superior a 60%. Acuracia obtida: " ++ show accuracy) (accuracy > 0.6))

testMnistAcceptable :: Test
testMnistAcceptable = TestCase (assertBool ("O modelo nao obteve precisao superior a 70%. Acuracia obtida: " ++ show accuracy) (accuracy > 0.7))


testMnistMinimumTest :: Test
testMnistMinimumTest = TestCase (assertBool ("O modelo nao obteve precisao superior a 10%. Acuracia obtida: " ++ show accuracyTest) (accuracyTest > 0.1))

testMnistLowTest :: Test
testMnistLowTest = TestCase (assertBool ("O modelo nao obteve precisao superior a 20%. Acuracia obtida: " ++ show accuracyTest) (accuracyTest > 0.2))

testMnistMediumTest :: Test
testMnistMediumTest = TestCase (assertBool ("O modelo nao obteve precisao superior a 40%. Acuracia obtida: " ++ show accuracyTest) (accuracyTest > 0.4))

testMnistHighTest :: Test
testMnistHighTest = TestCase (assertBool ("O modelo nao obteve precisao superior a 60%. Acuracia obtida: " ++ show accuracyTest) (accuracyTest > 0.6))

testMnistAcceptableTest :: Test
testMnistAcceptableTest = TestCase (assertBool ("O modelo nao obteve precisao superior a 70%. Acuracia obtida: " ++ show accuracyTest) (accuracyTest > 0.7))


testMnist :: Test
testMnist = TestList [ TestLabel "Verificacao de Acuracia Minima - Dataset MNIST"    testMnistMinimum
                     , TestLabel "Verificacao de Acuracia Baixa - Dataset MNIST"     testMnistLow
                     , TestLabel "Verificacao de Acuracia Media - Dataset MNIST"     testMnistMedium
                     , TestLabel "Verificacao de Acuracia Alta - Dataset MNIST"      testMnistHigh
                     , TestLabel "Verificacao de Acuracia Aceitavel - Dataset MNIST" testMnistAcceptable
                     , TestLabel "Verificacao de Acuracia Minima - Dataset MNIST (Test)"    testMnistMinimumTest
                     , TestLabel "Verificacao de Acuracia Baixa - Dataset MNIST (Test)"     testMnistLowTest
                     , TestLabel "Verificacao de Acuracia Media - Dataset MNIST (Test)"     testMnistMediumTest
                     , TestLabel "Verificacao de Acuracia Alta - Dataset MNIST (Test)"      testMnistHighTest
                     , TestLabel "Verificacao de Acuracia Aceitavel - Dataset MNIST (Test)" testMnistAcceptableTest
                     ]